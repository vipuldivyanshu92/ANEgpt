#!/usr/bin/env python3
"""
ane_train.py — Train a small GPT model on Apple Neural Engine

Uses reverse-engineered ANE private APIs via ane_bridge.py to run
transformer forward/backward passes on the Neural Engine.

Key design: Weights are baked into compiled ANE kernels, so we use gradient
accumulation to amortize compilation cost. Each "batch" compiles kernels once,
runs N accumulation steps with those fixed kernels, then updates weights.

Architecture matches nanochat's GPT but uses:
- ANE for all matrix multiplications (forward + backward dx)
- CPU for RMSNorm, attention scores, loss, dW gradients, Adam

Usage:
    python -m scripts.ane_train --depth=1 --dim=64 --heads=2 --seq-len=32
"""

import os
import sys
import time
import json
import struct
import argparse
import numpy as np
from pathlib import Path

CKPT_PATH = os.path.join(os.environ.get('NANOCHAT_BASE_DIR', os.path.expanduser('~/.cache/nanochat')), 'ane_ckpt.npz')

sys.path.insert(0, str(Path(__file__).parent.parent))
from nanochat.ane_bridge import ANEBridge, ANEKernel, generate_conv_mil


# --- Math helpers (CPU) ---

def rmsnorm(x, w, eps=1e-5):
    """RMSNorm: x * w / sqrt(mean(x^2) + eps). x: [C, S], w: [C]"""
    rms = np.sqrt(np.mean(x * x, axis=0, keepdims=True) + eps)
    return x * w.reshape(-1, 1) / rms

def rmsnorm_backward(dy, x, w, eps=1e-5):
    """Backward through RMSNorm. Returns dx, dw."""
    C = x.shape[0]
    rms = np.sqrt(np.mean(x * x, axis=0, keepdims=True) + eps)
    xnorm = x / rms
    dw = np.sum(dy * xnorm, axis=1)
    wx = w.reshape(-1, 1) / rms
    dx = dy * wx
    mean_xy = np.mean(x * dy * w.reshape(-1, 1), axis=0, keepdims=True)
    dx = dx - x * mean_xy / (C * rms * rms)
    return dx, dw

def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)

def cross_entropy_loss(logits, targets):
    """logits: [V, S], targets: [S] int."""
    S = targets.shape[0]
    probs = softmax(logits, axis=0)
    loss = -np.sum(np.log(probs[targets, np.arange(S)] + 1e-10)) / S
    dlogits = probs.copy()
    dlogits[targets, np.arange(S)] -= 1.0
    return loss, dlogits / S


class CachedKernel:
    """A compiled ANE kernel that can be reused for multiple forward passes."""
    def __init__(self, kernel, out_shape):
        self.kernel = kernel
        self.out_shape = out_shape
    
    def run(self, x):
        """Run the kernel with new input data, reusing the same compiled program."""
        self.kernel.write_input(0, np.ascontiguousarray(x, dtype=np.float32))
        self.kernel.eval()
        return self.kernel.read_output(0, self.out_shape, np.float32)
    
    def free(self):
        if self.kernel:
            self.kernel.free()
            self.kernel = None


class ANEGPTTrainer:
    """Trains a GPT model using ANE for linear layer forward/backward."""
    
    def __init__(self, depth, dim, heads, seq_len, vocab_size, lr=1e-4, accum_steps=10):
        self.depth = depth
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.lr = lr
        self.hidden = 4 * dim
        self.accum_steps = accum_steps
        
        self.bridge = ANEBridge()
        self.bridge.init()
        
        self._init_weights()
        
        # Adam state
        self.adam_t = 0
        self.adam_m = {}
        self.adam_v = {}
        for name in self._weight_names():
            w = self._get_weight(name)
            self.adam_m[name] = np.zeros_like(w)
            self.adam_v[name] = np.zeros_like(w)

        # Cumulative stats (persisted across restarts)
        self.cum_steps = 0
        self.cum_compile_ms = 0.0
        self.cum_train_ms = 0.0
        self.cum_wall = 0.0
        self.last_loss = float('inf')
        self.initial_loss = None

    def save_checkpoint(self, path, batch_idx, total_batches, all_losses):
        """Save full trainer state for exec() restart."""
        d = {
            'batch_idx': np.array(batch_idx),
            'total_batches': np.array(total_batches),
            'adam_t': np.array(self.adam_t),
            'cum_steps': np.array(self.cum_steps),
            'cum_compile_ms': np.array(self.cum_compile_ms),
            'cum_train_ms': np.array(self.cum_train_ms),
            'cum_wall': np.array(self.cum_wall),
            'last_loss': np.array(self.last_loss),
            'embed': self.embed,
            'rms_final': self.rms_final,
        }
        if self.initial_loss is not None:
            d['initial_loss'] = np.array(self.initial_loss)
        if all_losses:
            d['all_losses'] = np.array(all_losses)
        for i in range(self.depth):
            for k, v in self.layers[i].items():
                d[f'layer{i}.{k}'] = v
        for name in self._weight_names():
            d[f'adam_m.{name}'] = self.adam_m[name]
            d[f'adam_v.{name}'] = self.adam_v[name]
        np.savez(path, **d)

    def load_checkpoint(self, path):
        """Load trainer state from checkpoint. Returns (batch_idx, total_batches, all_losses)."""
        d = np.load(path, allow_pickle=False)
        self.adam_t = int(d['adam_t'])
        self.cum_steps = int(d['cum_steps'])
        self.cum_compile_ms = float(d['cum_compile_ms'])
        self.cum_train_ms = float(d['cum_train_ms'])
        self.cum_wall = float(d['cum_wall'])
        self.last_loss = float(d['last_loss'])
        self.embed = d['embed'].copy()
        self.rms_final = d['rms_final'].copy()
        if 'initial_loss' in d:
            self.initial_loss = float(d['initial_loss'])
        for i in range(self.depth):
            for k in self.layers[i]:
                key = f'layer{i}.{k}'
                if key in d:
                    self.layers[i][k] = d[key].copy()
        for name in self._weight_names():
            mk = f'adam_m.{name}'
            vk = f'adam_v.{name}'
            if mk in d: self.adam_m[name] = d[mk].copy()
            if vk in d: self.adam_v[name] = d[vk].copy()
        batch_idx = int(d['batch_idx'])
        total_batches = int(d['total_batches'])
        all_losses = list(d['all_losses']) if 'all_losses' in d else []
        return batch_idx, total_batches, all_losses
    
    def _init_weights(self):
        D, H, V = self.dim, self.hidden, self.vocab_size
        scale = 0.02
        self.embed = np.random.randn(V, D).astype(np.float32) * scale
        self.rms_final = np.ones(D, dtype=np.float32)
        
        self.layers = []
        for _ in range(self.depth):
            layer = {
                'rms_att': np.ones(D, dtype=np.float32),
                'Wq': np.random.randn(D, D).astype(np.float32) * scale,
                'Wk': np.random.randn(D, D).astype(np.float32) * scale,
                'Wv': np.random.randn(D, D).astype(np.float32) * scale,
                'Wo': np.random.randn(D, D).astype(np.float32) * scale,
                'rms_ffn': np.ones(D, dtype=np.float32),
                'W1': np.random.randn(H, D).astype(np.float32) * scale,
                'W2': np.random.randn(D, H).astype(np.float32) * scale,
                'W3': np.random.randn(H, D).astype(np.float32) * scale,
            }
            self.layers.append(layer)
    
    def _weight_names(self):
        names = ['embed', 'rms_final']
        for i in range(self.depth):
            for k in ['rms_att', 'Wq', 'Wk', 'Wv', 'Wo', 'rms_ffn', 'W1', 'W2', 'W3']:
                names.append(f'layer{i}.{k}')
        return names
    
    def _get_weight(self, name):
        if name == 'embed': return self.embed
        if name == 'rms_final': return self.rms_final
        parts = name.split('.')
        idx = int(parts[0].replace('layer', ''))
        return self.layers[idx][parts[1]]
    
    def _compile_kernel(self, W, in_ch, out_ch, S):
        """Compile an ANE kernel for matmul y = W @ x."""
        W = np.ascontiguousarray(W, dtype=np.float32)
        mil = generate_conv_mil(in_ch, out_ch, S)
        w_blob = self.bridge.build_weight_blob(W)
        in_bytes = in_ch * S * 4
        out_bytes = out_ch * S * 4
        kernel = self.bridge.compile(mil, w_blob, [in_bytes], [out_bytes])
        return CachedKernel(kernel, (out_ch, S))
    
    def _compile_fwd_kernels(self):
        """Compile only forward kernels for current weights."""
        D, H, S = self.dim, self.hidden, self.seq_len
        fwd_layers = []
        
        for i in range(self.depth):
            layer = self.layers[i]
            lk = {}
            lk['Wq_fwd'] = self._compile_kernel(layer['Wq'], D, D, S)
            lk['Wk_fwd'] = self._compile_kernel(layer['Wk'], D, D, S)
            lk['Wv_fwd'] = self._compile_kernel(layer['Wv'], D, D, S)
            lk['Wo_fwd'] = self._compile_kernel(layer['Wo'], D, D, S)
            lk['W1_fwd'] = self._compile_kernel(layer['W1'], D, H, S)
            lk['W2_fwd'] = self._compile_kernel(layer['W2'], H, D, S)
            lk['W3_fwd'] = self._compile_kernel(layer['W3'], D, H, S)
            fwd_layers.append(lk)
        
        return fwd_layers
    
    def _compile_bwd_kernels(self):
        """Compile only backward kernels (transposed weights) for current weights."""
        D, H, S = self.dim, self.hidden, self.seq_len
        bwd_layers = []
        
        for i in range(self.depth):
            layer = self.layers[i]
            lk = {}
            lk['Wo_bwd'] = self._compile_kernel(layer['Wo'].T.copy(), D, D, S)
            lk['Wq_bwd'] = self._compile_kernel(layer['Wq'].T.copy(), D, D, S)
            lk['Wk_bwd'] = self._compile_kernel(layer['Wk'].T.copy(), D, D, S)
            lk['Wv_bwd'] = self._compile_kernel(layer['Wv'].T.copy(), D, D, S)
            lk['W1_bwd'] = self._compile_kernel(layer['W1'].T.copy(), H, D, S)
            lk['W2_bwd'] = self._compile_kernel(layer['W2'].T.copy(), D, H, S)
            lk['W3_bwd'] = self._compile_kernel(layer['W3'].T.copy(), H, D, S)
            bwd_layers.append(lk)
        
        return bwd_layers
    
    def _free_layer_kernels(self, layer_list):
        """Free kernels from a list of per-layer kernel dicts."""
        for lk in layer_list:
            for k in lk.values():
                k.free()
    
    def _forward_attention(self, x, layer, lk):
        """Attention forward using pre-compiled kernels."""
        D, S = self.dim, x.shape[1]
        H, HD = self.heads, self.head_dim
        
        xnorm = rmsnorm(x, layer['rms_att'])
        Q = lk['Wq_fwd'].run(xnorm)
        K = lk['Wk_fwd'].run(xnorm)
        V = lk['Wv_fwd'].run(xnorm)
        
        Q = Q.reshape(H, HD, S)
        K = K.reshape(H, HD, S)
        V = V.reshape(H, HD, S)
        
        scale = 1.0 / np.sqrt(HD)
        scores = np.einsum('hds,hdp->hsp', Q, K) * scale
        mask = np.triu(np.full((S, S), -1e9), k=1)
        scores = scores + mask[np.newaxis, :, :]
        attn = softmax(scores, axis=-1)
        attn_out = np.einsum('hsp,hdp->hds', attn, V).reshape(D, S)
        
        o_out = lk['Wo_fwd'].run(attn_out)
        out = x + o_out
        
        cache = {'x': x, 'xnorm': xnorm, 'Q': Q, 'K': K, 'V': V,
                 'attn': attn, 'attn_out': attn_out}
        return out, cache
    
    def _forward_ffn(self, x, layer, lk):
        """FFN forward using pre-compiled kernels."""
        xnorm = rmsnorm(x, layer['rms_ffn'])
        h1 = lk['W1_fwd'].run(xnorm)
        h3 = lk['W3_fwd'].run(xnorm)
        silu = h1 * (1.0 / (1.0 + np.exp(-np.clip(h1, -20, 20))))
        h_gate = silu * h3
        ffn_out = lk['W2_fwd'].run(h_gate)
        out = x + ffn_out
        
        cache = {'x': x, 'xnorm': xnorm, 'h1': h1, 'h3': h3,
                 'silu': silu, 'h_gate': h_gate}
        return out, cache
    
    def _backward_ffn(self, dy, layer, lk, cache):
        """FFN backward returning dx and weight gradients."""
        dh_gate = lk['W2_bwd'].run(dy)
        dW2 = dy @ cache['h_gate'].T
        
        dsilu = dh_gate * cache['h3']
        dh3 = dh_gate * cache['silu']
        sigmoid_h1 = 1.0 / (1.0 + np.exp(-np.clip(cache['h1'], -20, 20)))
        dh1 = dsilu * (sigmoid_h1 + cache['h1'] * sigmoid_h1 * (1 - sigmoid_h1))
        
        dxnorm = lk['W1_bwd'].run(dh1) + lk['W3_bwd'].run(dh3)
        dW1 = dh1 @ cache['xnorm'].T
        dW3 = dh3 @ cache['xnorm'].T
        
        dx_rms, drms_ffn = rmsnorm_backward(dxnorm, cache['x'], layer['rms_ffn'])
        dx = dy + dx_rms
        
        return dx, {'W1': dW1, 'W2': dW2, 'W3': dW3, 'rms_ffn': drms_ffn}
    
    def _backward_attention(self, dy, layer, lk, cache):
        """Attention backward returning dx and weight gradients."""
        D, S = self.dim, dy.shape[1]
        H, HD = self.heads, self.head_dim
        
        dattn_out = lk['Wo_bwd'].run(dy)
        dWo = dy @ cache['attn_out'].T
        
        dattn_out = dattn_out.reshape(H, HD, S)
        Q, K, V, attn = cache['Q'], cache['K'], cache['V'], cache['attn']
        scale = 1.0 / np.sqrt(HD)
        
        dV = np.einsum('hsp,hds->hdp', attn, dattn_out)
        dattn = np.einsum('hds,hdp->hsp', dattn_out, V)
        dscores = attn * (dattn - np.sum(dattn * attn, axis=-1, keepdims=True)) * scale
        
        dQ = np.einsum('hsp,hdp->hds', dscores, K).reshape(D, S)
        dK = np.einsum('hsp,hds->hdp', dscores, Q).reshape(D, S)
        dV = dV.reshape(D, S)
        
        dxnorm = lk['Wq_bwd'].run(dQ) + lk['Wk_bwd'].run(dK) + lk['Wv_bwd'].run(dV)
        dWq = dQ @ cache['xnorm'].T
        dWk = dK @ cache['xnorm'].T
        dWv = dV @ cache['xnorm'].T
        
        dx_rms, drms_att = rmsnorm_backward(dxnorm, cache['x'], layer['rms_att'])
        dx = dy + dx_rms
        
        return dx, {'Wq': dWq, 'Wk': dWk, 'Wv': dWv, 'Wo': dWo, 'rms_att': drms_att}
    
    def _forward_pass(self, tokens, fwd_kernels):
        """Run forward pass, return loss and saved state for backward."""
        D, S, V = self.dim, self.seq_len, self.vocab_size
        x = self.embed[tokens[:S]].T.copy().astype(np.float32)
        target = tokens[1:S+1]
        
        attn_caches, ffn_caches = [], []
        for i in range(self.depth):
            lk = fwd_kernels[i]
            x, ac = self._forward_attention(x, self.layers[i], lk)
            attn_caches.append(ac)
            x, fc = self._forward_ffn(x, self.layers[i], lk)
            ffn_caches.append(fc)
        
        x_final = rmsnorm(x, self.rms_final)
        logits = self.embed @ x_final
        loss, dlogits = cross_entropy_loss(logits, target)
        
        saved = {
            'tokens': tokens, 'x_pre_final': x, 'x_final': x_final,
            'dlogits': dlogits, 'attn_caches': attn_caches, 'ffn_caches': ffn_caches,
        }
        return loss, saved
    
    def _backward_pass(self, saved, bwd_kernels):
        """Run backward pass using saved forward state."""
        tokens = saved['tokens']
        S = self.seq_len
        x = saved['x_pre_final']
        x_final = saved['x_final']
        dlogits = saved['dlogits']
        attn_caches = saved['attn_caches']
        ffn_caches = saved['ffn_caches']
        
        # Backward through classifier
        dx_final = self.embed.T @ dlogits
        dembed = dlogits @ x_final.T
        dx, drms_final = rmsnorm_backward(dx_final, x, self.rms_final)
        
        # Backward through layers
        layer_grads = [{} for _ in range(self.depth)]
        for i in range(self.depth - 1, -1, -1):
            lk = bwd_kernels[i]
            dx, fg = self._backward_ffn(dx, self.layers[i], lk, ffn_caches[i])
            dx, ag = self._backward_attention(dx, self.layers[i], lk, attn_caches[i])
            layer_grads[i].update(fg)
            layer_grads[i].update(ag)
        
        # Embedding gradient
        S_len = tokens[:S].shape[0]
        for t in range(S_len):
            dembed[tokens[t]] += dx[:, t]
        
        return dembed, drms_final, layer_grads
    
    def train_batch(self, data, step_offset):
        """
        Train one batch with gradient accumulation.
        
        Split into two phases to halve peak ANE program loads:
        Phase 1: Compile forward kernels, run all accum_steps forward, free fwd kernels
        Phase 2: Compile backward kernels, run all accum_steps backward, free bwd kernels
        """
        S = self.seq_len
        max_pos = len(data) - S - 1
        
        # --- Phase 1: Forward pass (compile fwd kernels only) ---
        t_compile = time.time()
        try:
            fwd_kernels = self._compile_fwd_kernels()
        except RuntimeError as e:
            if 'ANE compile failed' in str(e):
                return None, 0, 0, []  # Signal caller to restart
            raise
        
        # Run forward for all accum_steps, saving state for backward
        total_loss = 0.0
        step_losses = []
        all_saved = []
        
        t_train = time.time()
        for s in range(self.accum_steps):
            pos = np.random.randint(0, max_pos)
            tokens = data[pos:pos + S + 1].astype(np.int64)
            loss, saved = self._forward_pass(tokens, fwd_kernels)
            total_loss += loss
            step_losses.append(loss)
            all_saved.append(saved)
        
        # Free forward kernels before compiling backward
        self._free_layer_kernels(fwd_kernels)
        
        # --- Phase 2: Backward pass (compile bwd kernels only) ---
        try:
            bwd_kernels = self._compile_bwd_kernels()
        except RuntimeError as e:
            if 'ANE compile failed' in str(e):
                return None, 0, 0, []  # Signal caller to restart
            raise
        compile_ms = (time.time() - t_compile) * 1000
        
        # Accumulate gradients from backward passes
        acc_dembed = np.zeros_like(self.embed)
        acc_drms_final = np.zeros_like(self.rms_final)
        acc_layer_grads = [{k: np.zeros_like(v) for k, v in self.layers[i].items() 
                           if k.startswith(('W', 'rms'))}
                          for i in range(self.depth)]
        
        for s in range(self.accum_steps):
            dembed, drms_final, layer_grads = self._backward_pass(all_saved[s], bwd_kernels)
            acc_dembed += dembed
            acc_drms_final += drms_final
            for i in range(self.depth):
                for k, g in layer_grads[i].items():
                    acc_layer_grads[i][k] += g
        
        train_ms = (time.time() - t_train) * 1000
        
        # Free backward kernels
        self._free_layer_kernels(bwd_kernels)
        del all_saved  # Free cached forward state
        
        # Average gradients
        n = self.accum_steps
        acc_dembed /= n
        acc_drms_final /= n
        for i in range(self.depth):
            for k in acc_layer_grads[i]:
                acc_layer_grads[i][k] /= n
        
        # Adam update
        self.adam_t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        
        def adam_update(w, g, name):
            self.adam_m[name] = b1 * self.adam_m[name] + (1 - b1) * g
            self.adam_v[name] = b2 * self.adam_v[name] + (1 - b2) * g * g
            m_hat = self.adam_m[name] / (1 - b1 ** self.adam_t)
            v_hat = self.adam_v[name] / (1 - b2 ** self.adam_t)
            return (w - self.lr * m_hat / (np.sqrt(v_hat) + eps)).astype(np.float32)
        
        self.embed = adam_update(self.embed, acc_dembed, 'embed')
        self.rms_final = adam_update(self.rms_final, acc_drms_final, 'rms_final')
        
        for i in range(self.depth):
            for k, g in acc_layer_grads[i].items():
                name = f'layer{i}.{k}'
                self.layers[i][k] = adam_update(self.layers[i][k], g, name)
        
        avg_loss = total_loss / n
        return avg_loss, compile_ms, train_ms, step_losses


def main():
    parser = argparse.ArgumentParser(description="Train GPT on Apple Neural Engine")
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--heads', type=int, default=2)
    parser.add_argument('--seq-len', type=int, default=32)
    parser.add_argument('--vocab-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--num-batches', type=int, default=6, help='Number of compile-batches')
    parser.add_argument('--accum-steps', type=int, default=10, help='Gradient accumulation steps per batch')
    parser.add_argument('--data-path', type=str, default=None)
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint (used internally by exec restart)')
    parser.add_argument('--ane-retries', type=int, default=0, help='ANE restart retry counter (internal)')
    args = parser.parse_args()
    
    # Compile budget: we now properly free ANE kernels, so lifetime limit is high.
    # We still restart the process occasionally (e.g. every ~500 compiles) to be
    # completely safe against any other memory leaks, but it's no longer a hard bottleneck.
    compiles_per_batch = args.depth * 14
    max_batches_per_exec = max(1, 500 // compiles_per_batch)

    trainer = ANEGPTTrainer(
        depth=args.depth, dim=args.dim, heads=args.heads,
        seq_len=args.seq_len, vocab_size=args.vocab_size,
        lr=args.lr, accum_steps=args.accum_steps
    )

    # Resume from checkpoint if restarting
    start_batch = 0
    all_losses = []
    if args.resume and os.path.exists(CKPT_PATH):
        start_batch, saved_total, saved_losses = trainer.load_checkpoint(CKPT_PATH)
        all_losses = saved_losses
        args.num_batches = saved_total
        print(f"  [RESUMED batch {start_batch}/{args.num_batches}, "
              f"loss={trainer.last_loss:.4f}, {trainer.cum_steps} steps done]")
    else:
        print("=" * 60)
        print("  ANE GPT Training — Apple Neural Engine")
        print("=" * 60)
        print(f"  depth={args.depth}, dim={args.dim}, heads={args.heads}")
        print(f"  seq_len={args.seq_len}, vocab_size={args.vocab_size}")
        print(f"  lr={args.lr}")
        print(f"  batches={args.num_batches}, accum_steps={args.accum_steps}")
        total_steps = args.num_batches * args.accum_steps
        total_compiles = args.num_batches * compiles_per_batch
        print(f"  total_steps={total_steps}, compiles/batch={compiles_per_batch}")
        print(f"  total_compiles={total_compiles} (budget: ~55 per exec(), will auto-restart)")
        # Clean up any stale checkpoint
        if os.path.exists(CKPT_PATH):
            os.remove(CKPT_PATH)

    n_params = sum(w.size for w in [trainer.embed, trainer.rms_final] +
                   [trainer.layers[i][k] for i in range(args.depth)
                    for k in trainer.layers[i]])
    if start_batch == 0:
        print(f"\n  Parameters: {n_params:,} ({n_params*4/1024/1024:.1f} MB fp32)")

    if args.data_path and os.path.exists(args.data_path):
        if start_batch == 0:
            print(f"  Data: {args.data_path}")
        data = np.fromfile(args.data_path, dtype=np.uint16)
    else:
        if start_batch == 0:
            print("  Data: synthetic random bytes")
        total_steps = args.num_batches * args.accum_steps
        data = np.random.randint(0, args.vocab_size,
                                 size=max(args.seq_len * total_steps * 2, 10000),
                                 dtype=np.uint16)

    if start_batch == 0:
        print(f"  Data tokens: {len(data):,}")
        print()

    t_start = time.time()
    global_step = start_batch * args.accum_steps
    batches_this_exec = 0

    for batch in range(start_batch, args.num_batches):
        # Check if we need to restart before this batch (fresh ANE compile budget)
        if batches_this_exec >= max_batches_per_exec:
            import subprocess
            wall_this = (time.time() - t_start) * 1000
            trainer.cum_wall += wall_this
            trainer.save_checkpoint(CKPT_PATH, batch, args.num_batches, all_losses)
            print(f"  [restart at batch {batch}, {trainer.bridge.compile_count} compiles]")
            sys.stdout.flush()
            time.sleep(3)  # Cooldown: let ANE daemon reclaim system-wide program slots
            # Spawn new process (fresh ANE compile budget) using venv Python
            nanochat_dir = str(Path(__file__).resolve().parent.parent)
            venv_python = os.path.join(nanochat_dir, '.venv', 'bin', 'python')
            if not os.path.exists(venv_python):
                venv_python = sys.executable
            cmd = [venv_python, '-m', 'scripts.ane_train',
                f'--depth={args.depth}', f'--dim={args.dim}', f'--heads={args.heads}',
                f'--seq-len={args.seq_len}', f'--vocab-size={args.vocab_size}',
                f'--lr={args.lr}', f'--num-batches={args.num_batches}',
                f'--accum-steps={args.accum_steps}',
                '--resume', '--ane-retries=0'] + ([f'--data-path={args.data_path}'] if args.data_path else [])
            result = subprocess.run(cmd, cwd=nanochat_dir)
            sys.exit(result.returncode)

        result = trainer.train_batch(data, global_step)
        avg_loss, compile_ms, train_ms, step_losses = result

        # ANE compile failed — force restart into a fresh process
        if avg_loss is None:
            max_ane_retries = 5
            if args.ane_retries >= max_ane_retries:
                print(f"  [FATAL: ANE load failed {max_ane_retries} times in a row, giving up]")
                print(f"  The ANE program slot pool is exhausted system-wide.")
                print(f"  Wait a few minutes and try again, or reduce --depth.")
                sys.exit(1)
            import subprocess
            wall_this = (time.time() - t_start) * 1000
            trainer.cum_wall += wall_this
            trainer.save_checkpoint(CKPT_PATH, batch, args.num_batches, all_losses)
            retry_num = args.ane_retries + 1
            cooldown = min(3 * retry_num, 15)  # Escalating cooldown: 3s, 6s, 9s, 12s, 15s
            print(f"  [ANE load failed at batch {batch}, retry {retry_num}/{max_ane_retries}, cooldown {cooldown}s]")
            sys.stdout.flush()
            time.sleep(cooldown)
            nanochat_dir = str(Path(__file__).resolve().parent.parent)
            venv_python = os.path.join(nanochat_dir, '.venv', 'bin', 'python')
            if not os.path.exists(venv_python):
                venv_python = sys.executable
            cmd = [venv_python, '-m', 'scripts.ane_train',
                f'--depth={args.depth}', f'--dim={args.dim}', f'--heads={args.heads}',
                f'--seq-len={args.seq_len}', f'--vocab-size={args.vocab_size}',
                f'--lr={args.lr}', f'--num-batches={args.num_batches}',
                f'--accum-steps={args.accum_steps}',
                f'--resume', f'--ane-retries={retry_num}'] + ([f'--data-path={args.data_path}'] if args.data_path else [])
            r = subprocess.run(cmd, cwd=nanochat_dir)
            sys.exit(r.returncode)

        all_losses.extend(step_losses)
        global_step += args.accum_steps
        batches_this_exec += 1

        trainer.cum_steps = global_step
        trainer.cum_compile_ms += compile_ms
        trainer.cum_train_ms += train_ms
        trainer.last_loss = avg_loss
        if trainer.initial_loss is None:
            trainer.initial_loss = step_losses[0]

        ms_per_step = train_ms / args.accum_steps
        total_elapsed = trainer.cum_wall / 1000 + (time.time() - t_start)
        tok_per_sec = global_step * args.seq_len / total_elapsed if total_elapsed > 0 else 0

        print(f"  batch {batch:3d}  steps={global_step:4d}  loss={avg_loss:.4f}  "
              f"compile={compile_ms:.0f}ms  {ms_per_step:.1f}ms/step  "
              f"{tok_per_sec:.0f} tok/s  compiles={trainer.bridge.compile_count}")

    # Final stats
    wall_this = (time.time() - t_start) * 1000
    total_wall = (trainer.cum_wall + wall_this) / 1000
    print()
    print("=" * 60)
    print("  Training Complete")
    print("=" * 60)
    print(f"  Total steps:  {len(all_losses)}")
    print(f"  Initial loss: {all_losses[0]:.4f}" if all_losses else "  Initial loss: N/A")
    print(f"  Final loss:   {all_losses[-1]:.4f}" if all_losses else "  Final loss: N/A")
    print(f"  Wall time:    {total_wall:.1f}s")
    if all_losses:
        print(f"  Avg ms/step:  {total_wall*1000/len(all_losses):.0f}")
    print(f"  Compile time: {trainer.cum_compile_ms:.0f}ms")
    print("=" * 60)

    # Clean up checkpoint
    if os.path.exists(CKPT_PATH):
        os.remove(CKPT_PATH)


if __name__ == "__main__":
    main()
