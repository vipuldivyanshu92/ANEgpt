// mil_dynamic.h — MIL generators using dynamic matmul (weights via IOSurface)
// Instead of conv(const_weight, x), we use matmul(x, W) where both come from input.
// Input layout: [1, IC, 1, SP] fp32, SP = SEQ + total_weight_cols
// Activations in sp[0:SEQ], weight matrices packed sequentially in sp[SEQ:]
#pragma once
#include "io.h"

#define MIL_HDR \
    @"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, " \
    "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, " \
    "{\"coremltools-version\", \"9.0\"}})]\n{\n"

// Helper: generate a dynamic matmul within a MIL function
// Slices activation [1,ic,1,seq] and weight [1,ic,1,oc] from input, does matmul
// act_sp_off: spatial offset for activations (usually 0)
// w_sp_off: spatial offset for weight block
// Returns variable name of result [1,oc,1,seq] in fp16
static void gen_dyn_matmul(NSMutableString *m, const char *prefix,
                           int ic, int oc, int seq,
                           int act_sp_off, int w_sp_off,
                           const char *input_var) {
    // Slice activations
    [m appendFormat:@"        tensor<int32, [4]> %s_ba = const()[name=string(\"%s_ba\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", prefix, prefix, act_sp_off];
    [m appendFormat:@"        tensor<int32, [4]> %s_sa = const()[name=string(\"%s_sa\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", prefix, prefix, ic, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> %s_act = slice_by_size(x=%s,begin=%s_ba,size=%s_sa)[name=string(\"%s_act\")];\n", ic, seq, prefix, input_var, prefix, prefix, prefix];
    // Slice weight
    [m appendFormat:@"        tensor<int32, [4]> %s_bw = const()[name=string(\"%s_bw\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", prefix, prefix, w_sp_off];
    [m appendFormat:@"        tensor<int32, [4]> %s_sw = const()[name=string(\"%s_sw\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", prefix, prefix, ic, oc];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> %s_wt = slice_by_size(x=%s,begin=%s_bw,size=%s_sw)[name=string(\"%s_wt\")];\n", ic, oc, prefix, input_var, prefix, prefix, prefix];
    // Reshape act: [1,ic,1,seq] → [1,1,ic,seq] → transpose → [1,1,seq,ic]
    [m appendFormat:@"        tensor<int32, [4]> %s_ra = const()[name=string(\"%s_ra\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", prefix, prefix, ic, seq];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> %s_a2 = reshape(shape=%s_ra,x=%s_act)[name=string(\"%s_a2\")];\n", ic, seq, prefix, prefix, prefix, prefix];
    [m appendFormat:@"        tensor<int32, [4]> %s_pm = const()[name=string(\"%s_pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n", prefix, prefix];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> %s_a3 = transpose(perm=%s_pm,x=%s_a2)[name=string(\"%s_a3\")];\n", seq, ic, prefix, prefix, prefix, prefix];
    // Reshape weight: [1,ic,1,oc] → [1,1,ic,oc]
    [m appendFormat:@"        tensor<int32, [4]> %s_rw = const()[name=string(\"%s_rw\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", prefix, prefix, ic, oc];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> %s_W = reshape(shape=%s_rw,x=%s_wt)[name=string(\"%s_W\")];\n", ic, oc, prefix, prefix, prefix, prefix];
    // matmul: [1,1,seq,ic] @ [1,1,ic,oc] → [1,1,seq,oc]
    [m appendString:@"        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> %s_yh = matmul(transpose_x=bF,transpose_y=bF,x=%s_a3,y=%s_W)[name=string(\"%s_yh\")];\n", seq, oc, prefix, prefix, prefix, prefix];
    // Transpose back + reshape: [1,1,seq,oc] → [1,1,oc,seq] → [1,oc,1,seq]
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> %s_yt = transpose(perm=%s_pm,x=%s_yh)[name=string(\"%s_yt\")];\n", oc, seq, prefix, prefix, prefix, prefix];
    [m appendFormat:@"        tensor<int32, [4]> %s_ro = const()[name=string(\"%s_ro\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", prefix, prefix, oc, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> %s_y = reshape(shape=%s_ro,x=%s_yt)[name=string(\"%s_y\")];\n", oc, seq, prefix, prefix, prefix, prefix];
}

// ===== Dynamic matmul kernel: y = x @ W =====
// Input: [1, IC, 1, SEQ+OC] fp32 — act[0:SEQ] + W[SEQ:SEQ+OC]
// Output: [1, OC, 1, SEQ] fp32
static NSString *gen_dyn_matmul_mil(int ic, int oc, int seq) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    int sp = seq + oc;
    [m appendFormat:@"    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n", ic, sp];
    [m appendString:@"        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xh = cast(dtype=to16,x=x)[name=string(\"cin\")];\n", ic, sp];
    gen_dyn_matmul(m, "mm", ic, oc, seq, 0, seq, "xh");
    [m appendString:@"        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];\n"];
    [m appendFormat:@"        tensor<fp32, [1,%d,1,%d]> y = cast(dtype=to32,x=mm_y)[name=string(\"cout\")];\n", oc, seq];
    [m appendString:@"    } -> (y);\n}\n"];
    return m;
}

// ===== SDPA forward (dynamic weights) =====
// Replaces gen_sdpa_fwd_taps: RMSNorm done on CPU, this kernel does QKV matmul + SDPA + Wo matmul
// Input: [1, DIM, 1, SEQ + 4*DIM] fp32
//   sp[0:SEQ]           = xnorm (rmsnorm output, DIM channels)
//   sp[SEQ:SEQ+DIM]     = Wq[DIM,DIM]
//   sp[SEQ+DIM:SEQ+2D]  = Wk[DIM,DIM]
//   sp[SEQ+2D:SEQ+3D]   = Wv[DIM,DIM]
//   sp[SEQ+3D:SEQ+4D]   = Wo[DIM,DIM]
// Output: [1, 6*DIM, 1, SEQ] fp16 = concat(o_out, Q, K, V, attn_out, xnorm_pass)
// NOTE: mask is still a const weight (it doesn't change)
static NSString *gen_sdpa_fwd_dynamic(void) {
    float sc = 1.0f/sqrtf((float)HD);
    int w_total = 4*DIM;  // Wq+Wk+Wv+Wo
    int sp_in = SEQ + w_total;
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n", DIM, sp_in];
    // Cast to fp16
    [m appendString:@"        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xh = cast(dtype=to16,x=x)[name=string(\"cin\")];\n", DIM, sp_in];

    // Slice xnorm [1,DIM,1,SEQ]
    [m appendString:@"        tensor<int32, [4]> bx = const()[name=string(\"bx\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<int32, [4]> sx = const()[name=string(\"sx\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xn = slice_by_size(x=xh,begin=bx,size=sx)[name=string(\"xn\")];\n", DIM, SEQ];

    // Slice Wq [1,DIM,1,DIM]
    [m appendFormat:@"        tensor<int32, [4]> bq = const()[name=string(\"bq\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", SEQ];
    [m appendFormat:@"        tensor<int32, [4]> sw = const()[name=string(\"sw\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> Wq = slice_by_size(x=xh,begin=bq,size=sw)[name=string(\"Wq\")];\n", DIM, DIM];

    // Slice Wk
    [m appendFormat:@"        tensor<int32, [4]> bk = const()[name=string(\"bk\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", SEQ+DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> Wk = slice_by_size(x=xh,begin=bk,size=sw)[name=string(\"Wk\")];\n", DIM, DIM];

    // Slice Wv
    [m appendFormat:@"        tensor<int32, [4]> bv = const()[name=string(\"bv\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", SEQ+2*DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> Wv = slice_by_size(x=xh,begin=bv,size=sw)[name=string(\"Wv\")];\n", DIM, DIM];

    // Slice Wo
    [m appendFormat:@"        tensor<int32, [4]> bo = const()[name=string(\"bo\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", SEQ+3*DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> Wo = slice_by_size(x=xh,begin=bo,size=sw)[name=string(\"Wo\")];\n", DIM, DIM];

    // Reshape for matmul: [1,D,1,S] → [1,1,D,S] → [1,1,S,D]
    [m appendFormat:@"        tensor<int32, [4]> r2 = const()[name=string(\"r2\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", DIM, SEQ];
    [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> xn2 = reshape(shape=r2,x=xn)[name=string(\"xn2\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> xnt = transpose(perm=pm,x=xn2)[name=string(\"xnt\")];\n", SEQ, DIM];

    // Reshape weights: [1,D,1,D] → [1,1,D,D]
    [m appendFormat:@"        tensor<int32, [4]> rw = const()[name=string(\"rw\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", DIM, DIM];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> Wq2 = reshape(shape=rw,x=Wq)[name=string(\"Wq2\")];\n", DIM, DIM];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> Wk2 = reshape(shape=rw,x=Wk)[name=string(\"Wk2\")];\n", DIM, DIM];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> Wv2 = reshape(shape=rw,x=Wv)[name=string(\"Wv2\")];\n", DIM, DIM];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> Wo2 = reshape(shape=rw,x=Wo)[name=string(\"Wo2\")];\n", DIM, DIM];

    // QKV matmul: [1,1,S,D] @ [1,1,D,D] → [1,1,S,D]
    [m appendString:@"        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"];
    [m appendString:@"        bool bT = const()[name=string(\"bT\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> qm = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=Wq2)[name=string(\"qm\")];\n", SEQ, DIM];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> km = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=Wk2)[name=string(\"km\")];\n", SEQ, DIM];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> vm = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=Wv2)[name=string(\"vm\")];\n", SEQ, DIM];

    // Transpose back: [1,1,S,D] → [1,1,D,S] → reshape [1,D,1,S]
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> qt = transpose(perm=pm,x=qm)[name=string(\"qt\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> kt = transpose(perm=pm,x=km)[name=string(\"kt\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> vt = transpose(perm=pm,x=vm)[name=string(\"vt\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> os = const()[name=string(\"os\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> qf = reshape(shape=os,x=qt)[name=string(\"qf\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> kf = reshape(shape=os,x=kt)[name=string(\"kf\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> vf = reshape(shape=os,x=vt)[name=string(\"vf\")];\n", DIM, SEQ];

    // SDPA: reshape to heads, matmul, mask, softmax, matmul
    [m appendFormat:@"        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> q4 = reshape(shape=qsh,x=qf)[name=string(\"rq\")];\n", HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> q = transpose(perm=pm,x=q4)[name=string(\"tq\")];\n", HEADS, SEQ, HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k4 = reshape(shape=qsh,x=kf)[name=string(\"rk\")];\n", HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k = transpose(perm=pm,x=k4)[name=string(\"tk\")];\n", HEADS, SEQ, HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> v4 = reshape(shape=qsh,x=vf)[name=string(\"rv\")];\n", HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> v = transpose(perm=pm,x=v4)[name=string(\"tv\")];\n", HEADS, SEQ, HD];

    // Q @ K^T
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> sc1 = matmul(transpose_x=bF,transpose_y=bT,x=q,y=k)[name=string(\"mm1\")];\n", HEADS, SEQ, SEQ];
    [m appendFormat:@"        fp16 scv = const()[name=string(\"scv\"), val=fp16(%f)];\n", sc];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> sc2 = mul(x=sc1,y=scv)[name=string(\"scl\")];\n", HEADS, SEQ, SEQ];

    // Causal mask (still const — doesn't change)
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> cm = const()[name=string(\"cm\"), val=tensor<fp16, [1,1,%d,%d]>(BLOBFILE(path=string(\"@model_path/weights/mask.bin\"), offset=uint64(64)))];\n", SEQ, SEQ, SEQ, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> ms = add(x=sc2,y=cm)[name=string(\"msk\")];\n", HEADS, SEQ, SEQ];

    // Softmax
    [m appendString:@"        int32 sax = const()[name=string(\"sax\"), val=int32(-1)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> aw = softmax(axis=sax,x=ms)[name=string(\"sm\")];\n", HEADS, SEQ, SEQ];

    // scores @ V
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> a4 = matmul(transpose_x=bF,transpose_y=bF,x=aw,y=v)[name=string(\"mm2\")];\n", HEADS, SEQ, HD];

    // Reshape back to [1,DIM,1,SEQ]
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> at = transpose(perm=pm,x=a4)[name=string(\"ta\")];\n", HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> af = reshape(shape=os,x=at)[name=string(\"ra\")];\n", DIM, SEQ];

    // Wo matmul: af → [1,1,S,D] @ Wo[1,1,D,D] → [1,1,S,D] → [1,D,1,S]
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> af2 = reshape(shape=r2,x=af)[name=string(\"af2\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> aft = transpose(perm=pm,x=af2)[name=string(\"aft\")];\n", SEQ, DIM];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> om = matmul(transpose_x=bF,transpose_y=bF,x=aft,y=Wo2)[name=string(\"om\")];\n", SEQ, DIM];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> ot = transpose(perm=pm,x=om)[name=string(\"ot\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> oo = reshape(shape=os,x=ot)[name=string(\"oo\")];\n", DIM, SEQ];

    // Output: concat(o_out, qf, kf, vf, af, xn) — same as original for backward compatibility
    [m appendString:@"        int32 cax = const()[name=string(\"cax\"), val=int32(1)];\n"];
    [m appendString:@"        bool cid = const()[name=string(\"cid\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = concat(axis=cax,interleave=cid,values=(oo,qf,kf,vf,af,xn))[name=string(\"cat\")];\n", 6*DIM, SEQ];
    // Cast to fp32
    [m appendString:@"        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];\n"];
    [m appendFormat:@"        tensor<fp32, [1,%d,1,%d]> out32 = cast(dtype=to32,x=out)[name=string(\"cout\")];\n", 6*DIM, SEQ];
    [m appendString:@"    } -> (out32);\n}\n"];
    return m;
}

// ===== FFN forward (dynamic weights) =====
// RMSNorm on CPU. This kernel: xnorm @ W1 → SiLU, xnorm @ W3 → gate, gate*silu @ W2 → out
// Input: [1, DIM, 1, SEQ + HIDDEN + HIDDEN + DIM] fp32
//   sp[0:SEQ]                        = xnorm [DIM,SEQ]
//   sp[SEQ:SEQ+HIDDEN]               = W1[DIM,HIDDEN]
//   sp[SEQ+HIDDEN:SEQ+2*HIDDEN]      = W3[DIM,HIDDEN]
//   sp[SEQ+2*HIDDEN:SEQ+2*HIDDEN+DIM]= W2[HIDDEN→DIM] — but W2 is [DIM,HIDDEN], we need HIDDEN input channels
// PROBLEM: W2 has shape [DIM,HIDDEN] = HIDDEN input channels, but our kernel has DIM input channels.
// Solution: separate kernels for W1/W3 (DIM→HIDDEN) and W2 (HIDDEN→DIM)
// OR: do W1,W3 in one kernel, SiLU on CPU/ANE, W2 in another kernel.
// Simpler: 3 separate matmul kernels per FFN direction. But that's too many dispatches.
// Better: one kernel for W1+W3 (same input dim), CPU SiLU, one kernel for W2.

// FFN part 1: xnorm @ W1, xnorm @ W3 (both DIM→HIDDEN)
// Input: [1, DIM, 1, SEQ + 2*HIDDEN] fp32
//   sp[0:SEQ]                  = xnorm
//   sp[SEQ:SEQ+HIDDEN]         = W1[DIM,HIDDEN]
//   sp[SEQ+HIDDEN:SEQ+2*HIDDEN]= W3[DIM,HIDDEN]
// Output: [1, 2*HIDDEN, 1, SEQ] fp32 = concat(h1, h3)
static NSString *gen_ffn_w13_dynamic(void) {
    int sp_in = SEQ + 2*HIDDEN;
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n", DIM, sp_in];
    [m appendString:@"        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xh = cast(dtype=to16,x=x)[name=string(\"cin\")];\n", DIM, sp_in];

    // Slice xnorm
    [m appendString:@"        tensor<int32, [4]> bx = const()[name=string(\"bx\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<int32, [4]> sx = const()[name=string(\"sx\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xn = slice_by_size(x=xh,begin=bx,size=sx)[name=string(\"xn\")];\n", DIM, SEQ];

    // Slice W1
    [m appendFormat:@"        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", SEQ];
    [m appendFormat:@"        tensor<int32, [4]> s1 = const()[name=string(\"s1\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, HIDDEN];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> W1 = slice_by_size(x=xh,begin=b1,size=s1)[name=string(\"W1\")];\n", DIM, HIDDEN];

    // Slice W3
    [m appendFormat:@"        tensor<int32, [4]> b3 = const()[name=string(\"b3\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", SEQ+HIDDEN];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> W3 = slice_by_size(x=xh,begin=b3,size=s1)[name=string(\"W3\")];\n", DIM, HIDDEN];

    // Reshape for matmul
    [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
    [m appendFormat:@"        tensor<int32, [4]> rd = const()[name=string(\"rd\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> xn2 = reshape(shape=rd,x=xn)[name=string(\"xn2\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> xnt = transpose(perm=pm,x=xn2)[name=string(\"xnt\")];\n", SEQ, DIM];

    [m appendFormat:@"        tensor<int32, [4]> rw = const()[name=string(\"rw\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", DIM, HIDDEN];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> W12 = reshape(shape=rw,x=W1)[name=string(\"W12\")];\n", DIM, HIDDEN];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> W32 = reshape(shape=rw,x=W3)[name=string(\"W32\")];\n", DIM, HIDDEN];

    [m appendString:@"        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> h1m = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=W12)[name=string(\"h1m\")];\n", SEQ, HIDDEN];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> h3m = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=W32)[name=string(\"h3m\")];\n", SEQ, HIDDEN];

    // Transpose back
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> h1t = transpose(perm=pm,x=h1m)[name=string(\"h1t\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> h3t = transpose(perm=pm,x=h3m)[name=string(\"h3t\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> rh = const()[name=string(\"rh\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> h1 = reshape(shape=rh,x=h1t)[name=string(\"h1\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> h3 = reshape(shape=rh,x=h3t)[name=string(\"h3\")];\n", HIDDEN, SEQ];

    // SiLU + gate
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> sig = sigmoid(x=h1)[name=string(\"sg\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> silu = mul(x=h1,y=sig)[name=string(\"si\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> gate = mul(x=silu,y=h3)[name=string(\"gt\")];\n", HIDDEN, SEQ];

    // Concat output: (h1, h3, gate)
    [m appendString:@"        int32 cax = const()[name=string(\"cax\"), val=int32(1)];\n"];
    [m appendString:@"        bool cid = const()[name=string(\"cid\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = concat(axis=cax,interleave=cid,values=(h1,h3,gate))[name=string(\"cat\")];\n", 2*HIDDEN+HIDDEN, SEQ];
    [m appendString:@"        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];\n"];
    [m appendFormat:@"        tensor<fp32, [1,%d,1,%d]> out32 = cast(dtype=to32,x=out)[name=string(\"cout\")];\n", 3*HIDDEN, SEQ];
    [m appendString:@"    } -> (out32);\n}\n"];
    return m;
}

// FFN part 2: gate @ W2 (HIDDEN→DIM)
// Input: [1, HIDDEN, 1, SEQ + DIM] fp32
//   sp[0:SEQ]       = gate [HIDDEN,SEQ]
//   sp[SEQ:SEQ+DIM] = W2[HIDDEN,DIM]
// Output: [1, DIM, 1, SEQ] fp32
static NSString *gen_ffn_w2_dynamic(void) {
    int sp_in = SEQ + DIM;
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n", HIDDEN, sp_in];
    [m appendString:@"        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xh = cast(dtype=to16,x=x)[name=string(\"cin\")];\n", HIDDEN, sp_in];

    [m appendString:@"        tensor<int32, [4]> ba = const()[name=string(\"ba\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<int32, [4]> sa = const()[name=string(\"sa\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> act = slice_by_size(x=xh,begin=ba,size=sa)[name=string(\"act\")];\n", HIDDEN, SEQ];

    [m appendFormat:@"        tensor<int32, [4]> bw = const()[name=string(\"bw\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", SEQ];
    [m appendFormat:@"        tensor<int32, [4]> sw = const()[name=string(\"sw\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", HIDDEN, DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> W2 = slice_by_size(x=xh,begin=bw,size=sw)[name=string(\"W2\")];\n", HIDDEN, DIM];

    [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
    [m appendFormat:@"        tensor<int32, [4]> ra = const()[name=string(\"ra\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> a2 = reshape(shape=ra,x=act)[name=string(\"a2\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> at = transpose(perm=pm,x=a2)[name=string(\"at\")];\n", SEQ, HIDDEN];

    [m appendFormat:@"        tensor<int32, [4]> rw = const()[name=string(\"rw\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", HIDDEN, DIM];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> W22 = reshape(shape=rw,x=W2)[name=string(\"W22\")];\n", HIDDEN, DIM];

    [m appendString:@"        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> ym = matmul(transpose_x=bF,transpose_y=bF,x=at,y=W22)[name=string(\"ym\")];\n", SEQ, DIM];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> yt = transpose(perm=pm,x=ym)[name=string(\"yt\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> yr = reshape(shape=ro,x=yt)[name=string(\"yr\")];\n", DIM, SEQ];
    [m appendString:@"        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];\n"];
    [m appendFormat:@"        tensor<fp32, [1,%d,1,%d]> y = cast(dtype=to32,x=yr)[name=string(\"cout\")];\n", DIM, SEQ];
    [m appendString:@"    } -> (y);\n}\n"];
    return m;
}

// ===== FFN backward (dynamic weights) =====
// Input: [1, DIM+2*HIDDEN, 1, SEQ + HIDDEN + DIM + DIM] fp32
// Actually simpler to split into separate backward kernels like forward.

// FFN backward part 1: dffn @ W2^T → dsilu (HIDDEN), then SiLU derivative
// Input: [1, DIM, 1, SEQ + HIDDEN] fp32
//   sp[0:SEQ]        = dffn [DIM, SEQ]
//   sp[SEQ:SEQ+HIDDEN]= W2^T [DIM, HIDDEN]
// Output: [1, HIDDEN, 1, SEQ] fp32 = dsilu_raw
static NSString *gen_ffn_bwd_w2t_dynamic(void) {
    return gen_dyn_matmul_mil(DIM, HIDDEN, SEQ);
}

// FFN backward part 2: dh1 @ W1^T + dh3 @ W3^T → dx
// We need h1,h3 for SiLU derivative, but those are on CPU.
// Actually the SiLU derivative + gating is element-wise, do on CPU.
// Then: dh1 @ W1^T and dh3 @ W3^T are two separate matmuls (HIDDEN→DIM).
// Combine into one kernel:
// Input: [1, HIDDEN, 1, SEQ + SEQ + DIM + DIM] fp32
//   sp[0:SEQ]              = dh1 [HIDDEN,SEQ]
//   sp[SEQ:2*SEQ]          = dh3 [HIDDEN,SEQ]
//   sp[2*SEQ:2*SEQ+DIM]    = W1^T [HIDDEN,DIM]
//   sp[2*SEQ+DIM:2*SEQ+2D] = W3^T [HIDDEN,DIM]
// Output: [1, DIM, 1, SEQ] fp32 = dx1 + dx3
static NSString *gen_ffn_bwd_w13t_dynamic(void) {
    int sp_in = 2*SEQ + 2*DIM;
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n", HIDDEN, sp_in];
    [m appendString:@"        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xh = cast(dtype=to16,x=x)[name=string(\"cin\")];\n", HIDDEN, sp_in];

    // Slice dh1 [HIDDEN, SEQ]
    [m appendString:@"        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<int32, [4]> sh = const()[name=string(\"sh\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dh1 = slice_by_size(x=xh,begin=b0,size=sh)[name=string(\"dh1\")];\n", HIDDEN, SEQ];

    // Slice dh3
    [m appendFormat:@"        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dh3 = slice_by_size(x=xh,begin=b1,size=sh)[name=string(\"dh3\")];\n", HIDDEN, SEQ];

    // Slice W1^T [HIDDEN, DIM]
    [m appendFormat:@"        tensor<int32, [4]> b2 = const()[name=string(\"b2\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", 2*SEQ];
    [m appendFormat:@"        tensor<int32, [4]> sw = const()[name=string(\"sw\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", HIDDEN, DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> W1t = slice_by_size(x=xh,begin=b2,size=sw)[name=string(\"W1t\")];\n", HIDDEN, DIM];

    // Slice W3^T
    [m appendFormat:@"        tensor<int32, [4]> b3 = const()[name=string(\"b3\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", 2*SEQ+DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> W3t = slice_by_size(x=xh,begin=b3,size=sw)[name=string(\"W3t\")];\n", HIDDEN, DIM];

    [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];

    // dh1 matmul: [S,H] @ [H,D] → [S,D]
    [m appendFormat:@"        tensor<int32, [4]> ra = const()[name=string(\"ra\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> dh12 = reshape(shape=ra,x=dh1)[name=string(\"dh12\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> dh1t = transpose(perm=pm,x=dh12)[name=string(\"dh1t\")];\n", SEQ, HIDDEN];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> dh32 = reshape(shape=ra,x=dh3)[name=string(\"dh32\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> dh3t = transpose(perm=pm,x=dh32)[name=string(\"dh3t\")];\n", SEQ, HIDDEN];

    [m appendFormat:@"        tensor<int32, [4]> rw = const()[name=string(\"rw\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", HIDDEN, DIM];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> W1t2 = reshape(shape=rw,x=W1t)[name=string(\"W1t2\")];\n", HIDDEN, DIM];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> W3t2 = reshape(shape=rw,x=W3t)[name=string(\"W3t2\")];\n", HIDDEN, DIM];

    [m appendString:@"        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> dx1m = matmul(transpose_x=bF,transpose_y=bF,x=dh1t,y=W1t2)[name=string(\"dx1m\")];\n", SEQ, DIM];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> dx3m = matmul(transpose_x=bF,transpose_y=bF,x=dh3t,y=W3t2)[name=string(\"dx3m\")];\n", SEQ, DIM];

    // Add
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> dxm = add(x=dx1m,y=dx3m)[name=string(\"dxm\")];\n", SEQ, DIM];

    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> dxt = transpose(perm=pm,x=dxm)[name=string(\"dxt\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dx = reshape(shape=ro,x=dxt)[name=string(\"dx\")];\n", DIM, SEQ];
    [m appendString:@"        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];\n"];
    [m appendFormat:@"        tensor<fp32, [1,%d,1,%d]> y = cast(dtype=to32,x=dx)[name=string(\"cout\")];\n", DIM, SEQ];
    [m appendString:@"    } -> (y);\n}\n"];
    return m;
}

// ===== SDPA backward part 1 (dynamic Wo^T) =====
// Same as original gen_sdpa_bwd1 but Wo^T comes from input instead of const
// Input: [1, 4*DIM, 1, SEQ + DIM] fp32 — Q,K,V,dx2 in channels, Wo^T in spatial
// Wait — channels must match for all data. Q,K,V are [DIM,SEQ], dx2 is [DIM,SEQ].
// Total input channels = 4*DIM. But Wo^T is [DIM,DIM] = DIM channels of DIM spatial.
// Problem: can't mix 4*DIM channels for data with DIM channels for Wo^T.
// Solution: Wo^T matmul as separate kernel, then SDPA part purely element-wise on ANE.

// Wo^T matmul: dx2 @ Wo^T → da (DIM→DIM)
static NSString *gen_wot_dynamic(void) {
    return gen_dyn_matmul_mil(DIM, DIM, SEQ);
}

// SDPA backward part 1 (no weights, all data): Q,K,V,da → dV,probs,dp
// Same as original but without Wo^T conv (already done)
// Input: [1, 4*DIM, 1, SEQ] fp16
static NSString *gen_sdpa_bwd1_noweight(void) {
    float sc = 1.0f/sqrtf((float)HD);
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", 4*DIM, SEQ];

    // Slice Q,K,V,da
    [m appendFormat:@"        tensor<int32, [4]> sz = const()[name=string(\"sz\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
    [m appendString:@"        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> qf = slice_by_size(x=x,begin=b0,size=sz)[name=string(\"s0\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> kf = slice_by_size(x=x,begin=b1,size=sz)[name=string(\"s1\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b2 = const()[name=string(\"b2\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", 2*DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> vf = slice_by_size(x=x,begin=b2,size=sz)[name=string(\"s2\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b3 = const()[name=string(\"b3\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", 3*DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> da = slice_by_size(x=x,begin=b3,size=sz)[name=string(\"s3\")];\n", DIM, SEQ];

    // Reshape to heads
    [m appendFormat:@"        tensor<int32, [4]> rsh = const()[name=string(\"rsh\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", HEADS, HD, SEQ];
    [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> qr = reshape(shape=rsh,x=qf)[name=string(\"rq\")];\n", HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> q = transpose(perm=pm,x=qr)[name=string(\"tq\")];\n", HEADS, SEQ, HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> kr = reshape(shape=rsh,x=kf)[name=string(\"rk\")];\n", HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k = transpose(perm=pm,x=kr)[name=string(\"tk\")];\n", HEADS, SEQ, HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> vr = reshape(shape=rsh,x=vf)[name=string(\"rv\")];\n", HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> v = transpose(perm=pm,x=vr)[name=string(\"tv\")];\n", HEADS, SEQ, HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dr = reshape(shape=rsh,x=da)[name=string(\"rd\")];\n", HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dat = transpose(perm=pm,x=dr)[name=string(\"td\")];\n", HEADS, SEQ, HD];

    // Forward attention scores (recompute)
    [m appendString:@"        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"];
    [m appendString:@"        bool bT = const()[name=string(\"bT\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> sc1 = matmul(transpose_x=bF,transpose_y=bT,x=q,y=k)[name=string(\"mm1\")];\n", HEADS, SEQ, SEQ];
    [m appendFormat:@"        fp16 scv = const()[name=string(\"scv\"), val=fp16(%f)];\n", sc];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> sc2 = mul(x=sc1,y=scv)[name=string(\"scl\")];\n", HEADS, SEQ, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> cm = const()[name=string(\"cm\"), val=tensor<fp16, [1,1,%d,%d]>(BLOBFILE(path=string(\"@model_path/weights/mask.bin\"), offset=uint64(64)))];\n", SEQ, SEQ, SEQ, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> ms = add(x=sc2,y=cm)[name=string(\"msk\")];\n", HEADS, SEQ, SEQ];
    [m appendString:@"        int32 sax = const()[name=string(\"sax\"), val=int32(-1)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> probs = softmax(axis=sax,x=ms)[name=string(\"sm\")];\n", HEADS, SEQ, SEQ];

    // dV = probs^T @ da, dp = da @ V^T
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dv4 = matmul(transpose_x=bT,transpose_y=bF,x=probs,y=dat)[name=string(\"dv\")];\n", HEADS, SEQ, HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dp4 = matmul(transpose_x=bF,transpose_y=bT,x=dat,y=v)[name=string(\"dp\")];\n", HEADS, SEQ, SEQ];

    // Reshape dV back
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dvt = transpose(perm=pm,x=dv4)[name=string(\"dvt\")];\n", HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> dvs = const()[name=string(\"dvs\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dvf = reshape(shape=dvs,x=dvt)[name=string(\"dvf\")];\n", DIM, SEQ];

    // Flatten probs and dp for output
    [m appendFormat:@"        tensor<int32, [4]> scs = const()[name=string(\"scs\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", SCORE_CH, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> pf = reshape(shape=scs,x=probs)[name=string(\"pf\")];\n", SCORE_CH, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dpf = reshape(shape=scs,x=dp4)[name=string(\"dpf\")];\n", SCORE_CH, SEQ];

    [m appendString:@"        int32 cax = const()[name=string(\"cax\"), val=int32(1)];\n"];
    [m appendString:@"        bool cid = const()[name=string(\"cid\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = concat(axis=cax,interleave=cid,values=(dvf,pf,dpf))[name=string(\"cat\")];\n", DIM+2*SCORE_CH, SEQ];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// SDPA backward part 2: same as original (no weights, pure computation)
static NSString *gen_sdpa_bwd2(void) {
    float sc = 1.0f/sqrtf((float)HD);
    int bwd2_in = 2*SCORE_CH + 2*DIM;
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", bwd2_in, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> sz_sc = const()[name=string(\"szsc\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", SCORE_CH, SEQ];
    [m appendString:@"        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> pf = slice_by_size(x=x,begin=b0,size=sz_sc)[name=string(\"s0\")];\n", SCORE_CH, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", SCORE_CH];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dpf = slice_by_size(x=x,begin=b1,size=sz_sc)[name=string(\"s1\")];\n", SCORE_CH, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> sz_d = const()[name=string(\"szd\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b2 = const()[name=string(\"b2\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", 2*SCORE_CH];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> qf = slice_by_size(x=x,begin=b2,size=sz_d)[name=string(\"s2\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b3 = const()[name=string(\"b3\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", 2*SCORE_CH+DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> kf = slice_by_size(x=x,begin=b3,size=sz_d)[name=string(\"s3\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> ssh = const()[name=string(\"ssh\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", HEADS, SEQ, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> probs = reshape(shape=ssh,x=pf)[name=string(\"rp\")];\n", HEADS, SEQ, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dp = reshape(shape=ssh,x=dpf)[name=string(\"rdp\")];\n", HEADS, SEQ, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> rsh = const()[name=string(\"rsh\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", HEADS, HD, SEQ];
    [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> qr = reshape(shape=rsh,x=qf)[name=string(\"rq\")];\n", HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> q = transpose(perm=pm,x=qr)[name=string(\"tq\")];\n", HEADS, SEQ, HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> kr = reshape(shape=rsh,x=kf)[name=string(\"rk\")];\n", HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k = transpose(perm=pm,x=kr)[name=string(\"tk\")];\n", HEADS, SEQ, HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> pdp = mul(x=probs,y=dp)[name=string(\"pdp\")];\n", HEADS, SEQ, SEQ];
    [m appendString:@"        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([-1])];\n"];
    [m appendString:@"        bool kd = const()[name=string(\"kd\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,1]> spdp = reduce_sum(x=pdp,axes=rax,keep_dims=kd)[name=string(\"rs\")];\n", HEADS, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dps = sub(x=dp,y=spdp)[name=string(\"dps\")];\n", HEADS, SEQ, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> ds0 = mul(x=probs,y=dps)[name=string(\"ds0\")];\n", HEADS, SEQ, SEQ];
    [m appendFormat:@"        fp16 scv = const()[name=string(\"scv\"), val=fp16(%f)];\n", sc];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> ds = mul(x=ds0,y=scv)[name=string(\"ds\")];\n", HEADS, SEQ, SEQ];
    [m appendString:@"        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"];
    [m appendString:@"        bool bT = const()[name=string(\"bT\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dq4 = matmul(transpose_x=bF,transpose_y=bF,x=ds,y=k)[name=string(\"dq\")];\n", HEADS, SEQ, HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dk4 = matmul(transpose_x=bT,transpose_y=bF,x=ds,y=q)[name=string(\"dk\")];\n", HEADS, SEQ, HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dqt = transpose(perm=pm,x=dq4)[name=string(\"dqt\")];\n", HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dkt = transpose(perm=pm,x=dk4)[name=string(\"dkt\")];\n", HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> fs = const()[name=string(\"fs\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dqf = reshape(shape=fs,x=dqt)[name=string(\"dqf\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dkf = reshape(shape=fs,x=dkt)[name=string(\"dkf\")];\n", DIM, SEQ];
    [m appendString:@"        int32 cax = const()[name=string(\"cax\"), val=int32(1)];\n"];
    [m appendString:@"        bool cid = const()[name=string(\"cid\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = concat(axis=cax,interleave=cid,values=(dqf,dkf))[name=string(\"cat\")];\n", 2*DIM, SEQ];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// QKV backward (dynamic): dq @ Wq^T + dk @ Wk^T + dv @ Wv^T → dx
// Input: [1, DIM, 1, 3*SEQ + 3*DIM] fp32
//   sp[0:SEQ]              = dq  [DIM,SEQ]
//   sp[SEQ:2*SEQ]          = dk  [DIM,SEQ]
//   sp[2*SEQ:3*SEQ]        = dv  [DIM,SEQ]
//   sp[3*SEQ:3*SEQ+DIM]    = Wq^T [DIM,DIM]
//   sp[3*SEQ+DIM:3*SEQ+2D] = Wk^T [DIM,DIM]
//   sp[3*SEQ+2D:3*SEQ+3D]  = Wv^T [DIM,DIM]
// Output: [1, DIM, 1, SEQ] fp32 = dxq + dxk + dxv
static NSString *gen_qkvb_dynamic(void) {
    int sp_in = 3*SEQ + 3*DIM;
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n", DIM, sp_in];
    [m appendString:@"        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xh = cast(dtype=to16,x=x)[name=string(\"cin\")];\n", DIM, sp_in];

    // Slice dq, dk, dv
    [m appendFormat:@"        tensor<int32, [4]> sd = const()[name=string(\"sd\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
    [m appendString:@"        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dq = slice_by_size(x=xh,begin=b0,size=sd)[name=string(\"dq\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dk = slice_by_size(x=xh,begin=b1,size=sd)[name=string(\"dk\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b2 = const()[name=string(\"b2\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", 2*SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dv = slice_by_size(x=xh,begin=b2,size=sd)[name=string(\"dv\")];\n", DIM, SEQ];

    // Slice Wq^T, Wk^T, Wv^T
    [m appendFormat:@"        tensor<int32, [4]> sw = const()[name=string(\"sw\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, DIM];
    [m appendFormat:@"        tensor<int32, [4]> b3 = const()[name=string(\"b3\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", 3*SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> Wqt = slice_by_size(x=xh,begin=b3,size=sw)[name=string(\"Wqt\")];\n", DIM, DIM];
    [m appendFormat:@"        tensor<int32, [4]> b4 = const()[name=string(\"b4\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", 3*SEQ+DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> Wkt = slice_by_size(x=xh,begin=b4,size=sw)[name=string(\"Wkt\")];\n", DIM, DIM];
    [m appendFormat:@"        tensor<int32, [4]> b5 = const()[name=string(\"b5\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", 3*SEQ+2*DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> Wvt = slice_by_size(x=xh,begin=b5,size=sw)[name=string(\"Wvt\")];\n", DIM, DIM];

    [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
    [m appendString:@"        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"];

    // Reshape and matmul for each
    [m appendFormat:@"        tensor<int32, [4]> rd = const()[name=string(\"rd\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> rw = const()[name=string(\"rw\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", DIM, DIM];

    // dq @ Wq^T
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> dq2 = reshape(shape=rd,x=dq)[name=string(\"dq2\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> dqt = transpose(perm=pm,x=dq2)[name=string(\"dqt\")];\n", SEQ, DIM];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> Wqt2 = reshape(shape=rw,x=Wqt)[name=string(\"Wqt2\")];\n", DIM, DIM];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> dxq = matmul(transpose_x=bF,transpose_y=bF,x=dqt,y=Wqt2)[name=string(\"dxq\")];\n", SEQ, DIM];

    // dk @ Wk^T
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> dk2 = reshape(shape=rd,x=dk)[name=string(\"dk2\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> dkt = transpose(perm=pm,x=dk2)[name=string(\"dkt\")];\n", SEQ, DIM];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> Wkt2 = reshape(shape=rw,x=Wkt)[name=string(\"Wkt2\")];\n", DIM, DIM];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> dxk = matmul(transpose_x=bF,transpose_y=bF,x=dkt,y=Wkt2)[name=string(\"dxk\")];\n", SEQ, DIM];

    // dv @ Wv^T
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> dv2 = reshape(shape=rd,x=dv)[name=string(\"dv2\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> dvt = transpose(perm=pm,x=dv2)[name=string(\"dvt\")];\n", SEQ, DIM];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> Wvt2 = reshape(shape=rw,x=Wvt)[name=string(\"Wvt2\")];\n", DIM, DIM];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> dxv = matmul(transpose_x=bF,transpose_y=bF,x=dvt,y=Wvt2)[name=string(\"dxv\")];\n", SEQ, DIM];

    // Sum: dxq + dxk + dxv
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> dxqk = add(x=dxq,y=dxk)[name=string(\"aqk\")];\n", SEQ, DIM];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> dxall = add(x=dxqk,y=dxv)[name=string(\"aall\")];\n", SEQ, DIM];

    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> dxt = transpose(perm=pm,x=dxall)[name=string(\"dxt\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dx = reshape(shape=ro,x=dxt)[name=string(\"dx\")];\n", DIM, SEQ];
    [m appendString:@"        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];\n"];
    [m appendFormat:@"        tensor<fp32, [1,%d,1,%d]> y = cast(dtype=to32,x=dx)[name=string(\"cout\")];\n", DIM, SEQ];
    [m appendString:@"    } -> (y);\n}\n"];
    return m;
}

// Causal mask blob (used by sdpa_fwd and sdpa_bwd1)
static NSData *g_mask_blob = nil;
static NSData *get_mask_blob(void) {
    if (!g_mask_blob) {
        _Float16 *mask = (_Float16*)calloc(SEQ*SEQ, sizeof(_Float16));
        for(int t=0;t<SEQ;t++) for(int t2=0;t2<SEQ;t2++)
            mask[t*SEQ+t2] = (t2<=t) ? (_Float16)0.0f : (_Float16)(-65504.0f);
        g_mask_blob = build_blob_fp16(mask, SEQ*SEQ);
        free(mask);
    }
    return g_mask_blob;
}
