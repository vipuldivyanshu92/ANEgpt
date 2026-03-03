<!-- <img src="ane-training/training/dashboard.gif" width="720" alt="ANE training dashboard"> -->

# ANEgpt

**Train GPT models directly on Apple's Neural Engine.**

ANEgpt is an open-source project that runs transformer training — forward pass, backward pass, and weight updates — on the Apple Neural Engine (ANE) in Apple Silicon. No CoreML training APIs. No Metal. No GPU. Pure ANE compute, driven through reverse-engineered private APIs.

Apple does not expose any public API for training on the ANE. This project reverse-engineers the `_ANEClient` / `_ANECompiler` private APIs and the MIL (Model Intermediate Language) format to run custom compute graphs — including backpropagation — directly on ANE hardware.

---

## What This Does

- Constructs MIL (Model Intermediate Language) programs at runtime in Objective-C
- Compiles them in-memory to ANE programs via `_ANEInMemoryModelDescriptor` (no `.mlmodelc` on disk)
- Passes tensors via IOSurface shared memory in `[1, C, 1, S]` fp16 format
- Runs forward and backward dx passes on ANE; weight gradients (dW) on CPU via Accelerate cblas
- Includes Adam optimizer, gradient accumulation, and checkpoint/resume
- Stably manages ANE program memory by explicitly releasing Objective-C objects under ARC, enabling sustained training without compiler memory exhaustion

### What You Can Train

The main Obj-C training program (`train_large.m`) trains a **Stories110M** model — a 12-layer Llama2-architecture transformer (dim=768, hidden=2048, heads=12, seq=256, vocab=32000) on TinyStories data. This is hardcoded in `stories_config.h`.

The Python-based trainer (`ane_train.py`) trains a smaller configurable GPT model through the ANE bridge, using nanochat's infrastructure for tokenization and data loading.

---

## Architecture

### Training Loop — 6 ANE Kernel Types Per Layer

Each transformer layer uses 6 ANE kernel dispatches:

| Kernel | Function |
|--------|----------|
| `fwdAttn` | RMSNorm → QKV projection → SDPA → output projection |
| `fwdFFN` | RMSNorm → SwiGLU FFN (W1, W3, SiLU, W2) |
| `ffnBwd` | FFN backward (W2ᵀ → SiLU_bwd → W1ᵀ, W3ᵀ) |
| `sdpaBwd1` | Woᵀ → SDPA backward part 1 (dV, probs, dp) |
| `sdpaBwd2` | SDPA backward part 2 (softmax grad → dQ, dK) |
| `qkvBwd` | QKV backward (Wqᵀ + Wkᵀ + Wvᵀ → dx) |

For the 12-layer Stories110M model, this means **72 ANE kernels per compile** (60 weight-bearing + 12 weight-free sdpaBwd2).

**`train_large_ane.m`** extends this with 4 additional ANE kernel types (see [ANE-Offloaded Operations](#ane-offloaded-operations) below), bringing the total to **~86 kernels per compile**.

CPU handles: dW gradient accumulation (cblas_sgemm, runs async in parallel with ANE), Adam optimizer (weights must be mutated — impossible on ANE), NLL loss gradient (requires target token indexing).

### System Stack

```
┌───────────────────────────────────────────────┐
│              Python (ane_train.py)             │
│  GPT model · loss · optimizer · data pipeline  │
├───────────────────────────────────────────────┤
│            ane_bridge.py (ctypes)              │
├───────────────────────────────────────────────┤
│         libane_bridge.dylib (Obj-C)           │
│   _ANEInMemoryModelDescriptor · IOSurface     │
├───────────────────────────────────────────────┤
│          Apple Neural Engine (ANE)             │
│    MIL programs · fp16 · [1,C,1,S] layout     │
└───────────────────────────────────────────────┘
```

---

## Components

### `ane-training/` — ANE Runtime & Kernels (Obj-C)

The low-level engine. Reverse-engineers private `AppleNeuralEngine.framework` APIs to compile and run MIL programs on ANE. Modified after forking from [maderix/ane](https://github.com/maderix/ANE).

- **`training/train_large.m`** — Baseline training: 12-layer Stories110M, forward/backward, checkpoint, exec() restart
- **`training/train_large_ane.m`** — ANE-offloaded variant: moves classifier, softmax, RMSNorm backward to ANE
- **`training/ane_rmsnorm_bwd.h`** — MIL generator for RMSNorm backward on ANE
- **`training/ane_classifier.h`** — MIL generators for classifier, softmax, final RMSNorm on ANE
- **`training/stories_*.h`** — Config, IO, MIL generators, CPU ops
- **`inmem_*.m`, `sram_*.m`** — ANE benchmarks and hardware probes
- **`bridge/`** — C-callable shared library for Python access

### `nanochat/` — LLM Training Harness (Python)

Modified after forking fork of [Andrej Karpathy's nanochat](https://github.com/karpathy/nanochat). Covers tokenization, pretraining, SFT, RLHF, evaluation, and a ChatGPT-like web UI. Extended here with:

- **`scripts/ane_train.py`** — ANE training backend that routes linear layers through the ANE bridge
- **`runs/runane.sh`** — Script to build the bridge and run ANE training

---

## Performance

The code measures and prints performance metrics at runtime (ms/step, TFLOPS, ANE utilization %). When you run `train_large`, the efficiency report at the end shows these computed metrics.

**What the code computes** (from `train_large.m` lines 654–667): 
- Per-step timing breakdown: ANE eval, IO (fp16 conversion), classifier (cblas), cross-entropy, RMSNorm, cblas wait
- Sustained ANE TFLOPS = ANE FLOPs executed / train time
- ANE utilization % = sustained TFLOPS / 15.8 (Apple's published M4 ANE peak)

**Reported in `training/README.md`** for the 12-layer Stories110M config (dim=768, hidden=2048, seq=256):

| Component | Time (ms/step) |
|-----------|----------------|
| ANE eval | 9.6 |
| IO (fp16 conversion) | 4.1 |
| Classifier (cblas) | 9.1 |
| Cross-entropy + residuals | 14.4 |
| RMSNorm | 0.1 |
| **Total** | **107 ms/step** |

> **Note:** These numbers come from the upstream `ane-training` project and have not been independently verified by us. Your results will vary by hardware (M1/M2/M3/M4/M5) and macOS version.

### ANE-Offloaded Operations

`train_large_ane.m` moves additional operations from CPU to ANE, verified on M4/M5:

| Operation | CPU Time | ANE Time | Speedup | Kernel Type |
|---|---|---|---|---|
| Classifier forward (embed @ x) | 10.77 ms | 1.06 ms | **10.2×** | 32000-channel conv |
| Softmax over VOCAB=32000 | 81.11 ms | 2.40 ms | **33.8×** | MIL `softmax` op |
| RMSNorm backward (per layer) | 0.18 ms | 0.21 ms | ~1× | element-wise + reduce |
| Final RMSNorm | — | — | — | same as forward RMSNorm |

**What stays on CPU (and why):**
- **Adam optimizer** — impossible on ANE (weights are baked constants at compile time)
- **dW gradient accumulation** — already runs in parallel with ANE via GCD async dispatch
- **Classifier backward** — ANE rejects 32000-input-channel convs; matmul fallback is 2× slower than cblas
- **NLL loss + gradient** — requires per-position target token indexing (`gather`)

### `train_large` vs `train_large_ane`

| Operation | `train_large` (baseline) | `train_large_ane` (offloaded) |
|---|---|---|
| 12-layer forward (fwdAttn, fwdFFN) | ANE | ANE |
| 12-layer backward (ffnBwd, sdpaBwd, qkvBwd) | ANE | ANE |
| Final RMSNorm forward | CPU (vDSP) | **ANE** kernel |
| Classifier forward (`embed @ x`) | CPU (cblas) | **ANE** conv (10× faster) |
| Softmax over vocab=32000 | CPU (vDSP) | **ANE** softmax op (34× faster) |
| RMSNorm backward (per layer) | CPU (vDSP) | **ANE** kernel + CPU for dw |
| Classifier backward (`embed^T @ dlogits`) | CPU (cblas) | CPU (cblas) |
| dW gradient accumulation | CPU (async cblas) | CPU (async cblas) |
| Adam optimizer | CPU | CPU |
| NLL loss + gradient | CPU | CPU |

| Metric | `train_large` | `train_large_ane` |
|---|---|---|
| Kernels per compile | 72 | 86 |
| ms/step | ~106 | ~93 |

### Key Optimizations

- **Channel-first CPU layout** — matches ANE IOSurface `[1,C,1,S]` format, eliminates transpose overhead
- **NEON vectorized fp16↔fp32** — ARM NEON intrinsics for fast IOSurface data transfer
- **GCD async cblas overlap** — dW gradient sgemms run in parallel with ANE evals on a background dispatch queue
- **Deferred cblas wait** — wait pushed into next step's forward pass for overlap
- **Forward taps** — Q, K, V, attention scores exposed via concat outputs, avoiding CPU recompute
- **Split Kernel Lifecycle** — Forward and backward kernels are compiled and freed in separate phases, halving the peak number of concurrently loaded ANE programs
- **Strict ARC Memory Management** — ANE objects are explicitly nilled before struct deallocation in the C bridge to gracefully release system-wide resources and prevent `0x50004` load failures

---

## Hardware Probing Results

The `m5result.md` file documents actual hardware probing results from an **M5** (ANE H16 family, same as M4), run on 2026-03-01:

- **Weights are baked at compile time** — overwriting weight blobs and reloading does not change output. Recompilation is required when weights change.
- **QoS has no effect on ANE frequency** — all QoS values 0-63 produce identical latency (~0.07ms avg for a 256×256 conv)
- **`_ANEPerformanceStats`** has `hwExecutionTime` property for wall-clock ANE timing, but requires `perfStatsMask` to be set before eval
- **`_ANEChainingRequest`** exists with loopback support — could enable multi-layer execution without CPU round-trips (unexplored)

---

## Getting Started

### Requirements

- **macOS 15+** on Apple Silicon (tested on M4, M5)
- **Xcode Command Line Tools** (`xcode-select --install`)
- **Python 3.10+** (for data download and dashboard)
- [uv](https://docs.astral.sh/uv/) (for nanochat path only)

### Step 1: Download Training Data

The training programs require pretokenized [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) data (flat uint16 token IDs, Llama 2 BPE, 32K vocab). Run the included download script:

```bash
cd ane-training/training
bash download_data.sh
```

This downloads `data00.bin` (~41 MB, ~20M tokens) from [enio/TinyStories](https://huggingface.co/datasets/enio/TinyStories) on HuggingFace and saves it as `tinystories_data00.bin`.

> **Note:** The download fetches a ~993 MB tar.gz archive, extracts shard 0, then cleans up. Ensure you have enough temporary disk space.

### Step 2: (Optional) Download Pretrained Weights

Both training programs can start from pretrained [Stories110M](https://huggingface.co/karpathy/tinyllamas) weights (llama2.c format). Without them, training starts from random initialization.

```bash
mkdir -p assets/models    # from repo root
curl -L -o assets/models/stories110M.bin \
  https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin
```

### Step 3: Build & Train

#### Option A: Pure Obj-C training (ane-training)

```bash
cd ane-training/training

# Baseline training (classifier + softmax on CPU)
make train_large && ./train_large

# ANE-offloaded training (classifier + softmax on ANE — faster)
make train_large_ane && ./train_large_ane
```

**CLI flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--steps N` | 10000 | Total training steps |
| `--lr F` | 3e-4 | Learning rate |
| `--resume` | — | Resume from checkpoint |

```bash
# Quick 100-step test run
./train_large --steps 100

# Resume training from checkpoint
./train_large --resume
```

The training loop automatically handles the ANE ~119 compile limit by saving a checkpoint and `exec()` restarting — this is transparent and the process resumes from where it left off.

#### Option B: Python-based training (nanochat + ANE bridge)

```bash
cd nanochat
bash runs/runane.sh
```

This will set up the virtual environment, build the ANE bridge, and train a tiny model on synthetic data. See `runs/runane.sh` for details.

**Example Benchmark (Python wrapper)**:
```text
  Total steps:  200
  Wall time:    21.2s
  Avg ms/step:  106 ms
  Compile time: 14095 ms
```
*(Tested on an M-series chip with no process restarts via the stable ARC integration)*

### Step 4: Monitor with Dashboard

The TUI dashboard shows real-time loss curves, power/CPU/memory graphs, and text generation:

```bash
# Install dashboard dependencies
pip install blessed psutil numpy

# Run alongside training (in a separate terminal)
cd ane-training/training
sudo python3 dashboard.py          # live mode (needs powermetrics via sudo)
sudo python3 dashboard.py --resume  # attach to running/resumed training
```

### Step 5: Benchmarking

Both training programs print an **Efficiency Report** at the end of training with per-step timing and ANE utilization:

```
=== Efficiency Report ===
Total steps:     100
Wall time:       45231 ms (45.2 s)
Compile time:    8200 ms (18.1%)
Train time:      37031 ms (81.9%)
Avg train:       107.0 ms/step
ANE TFLOPS:      2.45 sustained
Total TFLOPS:    3.12 (ANE+CPU)
ANE utilization: 15.5% of 15.8 TFLOPS
```

**Per-batch breakdown** is also printed during training:

```
  [batch 10: compile=7580ms train=1070ms (107.0ms/step) compiles=72]
    ane=9.6 io=4.1 cls=9.1 elem=14.4 rms=0.1 cblas_wait=2.3 ms/step
```

| Metric | Description |
|--------|-------------|
| `ane` | ANE kernel evaluation time |
| `io` | NEON fp16↔fp32 IOSurface data transfer |
| `cls` | Classifier matmul (cblas) — only in `train_large` |
| `elem` | Embedding lookup, residual adds, cross-entropy |
| `rms` | RMSNorm forward/backward (CPU) |
| `cblas_wait` | Time waiting for async dW gradient sgemms |
| `ANE TFLOPS` | Sustained FLOPs on ANE / train time |
| `ANE utilization` | Sustained TFLOPS / 15.8 (Apple's published M4 ANE peak) |

To compare baseline vs ANE-offloaded performance:

```bash
# Baseline benchmark (100 steps)
make train_large && ./train_large --steps 100

# ANE-offloaded benchmark (100 steps)
make train_large_ane && ./train_large_ane --steps 100
```

The key difference: `train_large_ane` moves classifier forward (10×), softmax (34×), and RMSNorm backward to ANE, reducing the `cls` and `elem` components significantly.

---

## Known Limitations

- **Weights baked at compile time** — every weight update requires recompilation of all kernels (verified on M5, see `m5result.md`)
- **SDPA causal masking** — ANE hardware ignores `attn_mask` in SDPA ops; causal attention is decomposed into separate Q@Kᵀ (ANE) → mask+softmax → scores@V (ANE)
- **macOS only** — requires Apple Silicon and private framework APIs
- **Undocumented APIs** — may break with macOS updates

---

## TODO

- [ ] Multi-layer chaining via `_ANEChainingRequest` to reduce CPU round-trips between layers
- [ ] Explore `_ANEPerformanceStats.hwExecutionTime` for accurate ANE timing
- [ ] Real-time eval path (`evaluateRealTimeWithModel:`) for lower latency
- [ ] Higher accumulation steps to amortize compile cost
- [ ] Fuse residual `add` into forward kernels (eliminate CPU round-trip for skip connections)
- [ ] Tile classifier backward for ANE (32000-input-ch conv fails, matmul is slow)
- [ ] Integration with nanochat's SFT/RLHF stages on ANE
- [ ] Compatibility testing across Apple Silicon generations (M1/M2/M3/M4/M5)
- [ ] Document discovered MIL instructions and ANE behavior

---

## Acknowledgements

This project builds on the following work:

- **[maderix/ane](https://github.com/maderix/ane)** — The original reverse-engineering of ANE private APIs for neural network training. The `ane-training/` directory is based on this work.
- **[Andrej Karpathy / nanochat](https://github.com/karpathy/nanochat)** — The simplest full-stack LLM training harness. The `nanochat/` directory is a fork extended with ANE training support.
- **[KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)** — Gamified nanoGPT with leaderboards, which inspired nanochat's speedrun approach.

---

## Disclaimer

This project is independent research into Apple Neural Engine architecture. It uses undocumented APIs discovered through runtime introspection for **research and educational purposes** under fair use and interoperability provisions (see *Sega v. Accolade*, 1992; DMCA §1201(f)). No Apple proprietary code or binaries are included in this repository. This project is **not affiliated with or endorsed by Apple Inc.** Use at your own risk.

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

The included components also carry MIT licenses:
- `ane-training/` — MIT © 2026 maderix
- `nanochat/` — MIT © 2025 Andrej Karpathy

---

<p align="center">
  <i>Built with curiosity, reverse engineering, and a healthy disregard for "inference only."</i>
</p>
