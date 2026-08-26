// Depthwise 2D convolution, whcn layout -- the semantics of ggml's GGML_OP_CONV_2D_DW
// on the contiguous-input path (ggml/src/ggml-cuda/conv2d-dw.cu, whcn_layout).
//
// Scalars (IN_W, IN_H, CHANNELS, BATCHES, KERNEL_W, KERNEL_H, STRIDE, PADDING,
// DILATION, OUT_W, OUT_H) arrive as -D macros; the contract is pointer args only,
// in inputs.yaml `args` order.
//
// Tap order is ky ascending, then kx ascending, skipping out-of-range taps. ggml's
// kernel derives the same valid range analytically via calculate_kernel_bounds();
// the explicit test below visits exactly the same taps in exactly the same order, so
// the fp32 accumulation is bit-identical rather than merely close.

#ifdef __cplusplus
extern "C" {
#endif

void conv2d_dw_reference(const float* input, const float* filter, float* output) {
    const long in_plane  = (long)IN_W  * (long)IN_H;
    const long out_plane = (long)OUT_W * (long)OUT_H;

    for (int n = 0; n < BATCHES; ++n) {
        for (int c = 0; c < CHANNELS; ++c) {
            const float* in_c  = input  + ((long)n * CHANNELS + c) * in_plane;
            const float* flt_c = filter + (long)c * KERNEL_H * KERNEL_W;
            float*       out_c = output + ((long)n * CHANNELS + c) * out_plane;

            for (int oy = 0; oy < OUT_H; ++oy) {
                for (int ox = 0; ox < OUT_W; ++ox) {
                    float acc = 0.0f;

                    for (int ky = 0; ky < KERNEL_H; ++ky) {
                        const int iy = oy * STRIDE + ky * DILATION - PADDING;
                        if (iy < 0 || iy >= IN_H) continue;

                        for (int kx = 0; kx < KERNEL_W; ++kx) {
                            const int ix = ox * STRIDE + kx * DILATION - PADDING;
                            if (ix < 0 || ix >= IN_W) continue;

                            acc += in_c[(long)iy * IN_W + ix]
                                 * flt_c[(long)ky * KERNEL_W + kx];
                        }
                    }

                    out_c[(long)oy * OUT_W + ox] = acc;
                }
            }
        }
    }
}

#ifdef __cplusplus
}
#endif
