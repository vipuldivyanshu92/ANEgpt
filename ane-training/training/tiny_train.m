// tiny_train.m — Train a 2-layer linear model on ANE (forward AND backward)
// y = W2 @ relu(W1 @ x), MSE loss
// Now using dynamic weights packed into the IOSurface

#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>
#include <math.h>

#import "training_dynamic/mil_dynamic.h"

static void ane_eval_dyn(Kern *k, const float *in, const float *w, float *out, int in_ch, int out_ch, int sp, bool t_w) {
    int sp_total = sp + out_ch;
    float *tmp = (float*)malloc(in_ch * sp_total * sizeof(float));
    for (int t = 0; t < sp; t++) {
        for (int c = 0; c < in_ch; c++) {
            tmp[c*sp_total + t] = in[t*in_ch + c];
        }
    }
    // We want to pack w.T into the rest: w.T is [in_ch, out_ch]
    for (int c = 0; c < in_ch; c++) {
        for (int o = 0; o < out_ch; o++) {
            if (t_w) {
                tmp[c*sp_total + sp + o] = w[o*in_ch + c]; // already transposed
            } else {
                tmp[c*sp_total + sp + o] = w[o*in_ch + c]; // W is [out_ch, in_ch] in original arrays! Wait!
            }
        }
    }
    IOSurfaceLock(k->ioIn, 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(k->ioIn), tmp, in_ch * sp_total * sizeof(float));
    IOSurfaceUnlock(k->ioIn, 0, NULL);
    free(tmp);

    NSError *e = nil;
    ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)((__bridge id)k->model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, (__bridge id)k->request, &e);

    float *tmp2 = (float*)malloc(out_ch * sp * sizeof(float));
    IOSurfaceLock(k->ioOut, kIOSurfaceLockReadOnly, NULL);
    memcpy(tmp2, IOSurfaceGetBaseAddress(k->ioOut), out_ch * sp * sizeof(float));
    IOSurfaceUnlock(k->ioOut, kIOSurfaceLockReadOnly, NULL);
    for (int t = 0; t < sp; t++)
        for (int c = 0; c < out_ch; c++)
            out[t*out_ch + c] = tmp2[c*sp + t];
    free(tmp2);
}

static double tb_to_ms(uint64_t elapsed, mach_timebase_info_data_t tb) {
    return (double)elapsed * tb.numer / tb.denom / 1e6;
}
static mach_timebase_info_data_t g_tb;

int main() {
    @autoreleasepool {
        ane_init();
        mach_timebase_info(&g_tb);
        int D = 64, H = 128, S = 16, total_steps = 50, ACCUM = 10;
        float lr = 1.0f;
        
        float *W1 = (float*)malloc(H * D * sizeof(float));
        float *W2 = (float*)malloc(D * H * sizeof(float));
        for (int i = 0; i < H*D; i++) W1[i] = 0.01f * sinf(i * 1.3f + 0.7f);
        for (int i = 0; i < D*H; i++) W2[i] = 0.01f * cosf(i * 0.9f + 1.1f);
        
        float *x = (float*)calloc(S*D, 4); float *y_targ = (float*)calloc(S*D, 4);
        for(int i=0; i<S*D; i++) { x[i] = sinf(i*0.1f); y_targ[i] = x[i]; }
        
        Kern *k1_fwd = compile_kern_mil_w(gen_dyn_matmul_mil(D, H, S), @{}, D * (S + H) * 4, H * S * 4);
        Kern *k2_fwd = compile_kern_mil_w(gen_dyn_matmul_mil(H, D, S), @{}, H * (S + D) * 4, D * S * 4);
        Kern *k2_bwd = compile_kern_mil_w(gen_dyn_matmul_mil(D, H, S), @{}, D * (S + H) * 4, H * S * 4);
        Kern *k1_bwd = compile_kern_mil_w(gen_dyn_matmul_mil(H, D, S), @{}, H * (S + D) * 4, D * S * 4);
        
        float *h = (float*)malloc(S*H*sizeof(float));
        float *h_relu = (float*)malloc(S*H*sizeof(float));
        float *y = (float*)malloc(S*D*sizeof(float));
        float *dy = (float*)malloc(S*D*sizeof(float));
        float *dh_relu = (float*)malloc(S*H*sizeof(float));
        float *dh = (float*)malloc(S*H*sizeof(float));
        float *dx = (float*)malloc(S*D*sizeof(float));
        
        printf("Dynamic ANE Training (No exec restarts!)\n");
        uint64_t t0 = mach_absolute_time();
        
        int step = 0;
        while(step < total_steps) {
            float *aW1 = (float*)calloc(H*D, 4);
            float *aW2 = (float*)calloc(D*H, 4);
            float loss = 0;
            
            for(int a=0; a<ACCUM; a++, step++) {
                // fwd
                ane_eval_dyn(k1_fwd, x, W1, h, D, H, S, false);
                for(int i=0; i<S*H; i++) h_relu[i] = h[i]>0?h[i]:0;
                ane_eval_dyn(k2_fwd, h_relu, W2, y, H, D, S, false);
                
                loss = 0;
                for(int i=0; i<S*D; i++) {
                    float diff = y[i] - y_targ[i];
                    loss += diff*diff; dy[i] = 2.0f*diff/(S*D);
                }
                loss /= (S*D);
                
                // bwd
                ane_eval_dyn(k2_bwd, dy, W2, dh_relu, D, H, S, true); // transposed weight usage
                for(int i=0; i<S*H; i++) dh[i] = h[i]>0?dh_relu[i]:0;
                ane_eval_dyn(k1_bwd, dh, W1, dx, H, D, S, true);
                
                for(int t=0; t<S; t++)
                    for(int i=0; i<D; i++)
                        for(int j=0; j<H; j++)
                            aW2[i*H+j] += dy[t*D+i]*h_relu[t*H+j];
                for(int t=0; t<S; t++)
                    for(int i=0; i<H; i++)
                        for(int j=0; j<D; j++)
                            aW1[i*D+j] += dh[t*H+i]*x[t*D+j];
            }
            for(int i=0; i<H*D; i++) W1[i] -= lr * aW1[i] / ACCUM;
            for(int i=0; i<D*H; i++) W2[i] -= lr * aW2[i] / ACCUM;
            free(aW1); free(aW2);
            printf("Step %d - Loss: %.4f\n", step, loss);
        }
        double ms = tb_to_ms(mach_absolute_time() - t0, g_tb);
        printf("Time: %.1fms (%.1fms/step)\n", ms, ms/total_steps);
    }
    return 0;
}
