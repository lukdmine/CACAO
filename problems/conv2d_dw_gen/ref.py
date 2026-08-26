"""Depthwise 2D convolution reference -- the semantics of ggml's GGML_OP_CONV_2D_DW
on the contiguous-input (whcn) path.

whcn means W is the fastest-varying axis, so the flat buffer reshapes directly to
(N, C, H, W). The filter bank is (KW, KH, C) with KW fastest -> (C, KH, KW).

ggml *skips* out-of-range taps; F.conv2d *pads with zeros*. These agree exactly: a
skipped tap contributes nothing and a zero-padded tap contributes 0.0 * w == 0.0.

tf32 is disabled explicitly. It is on by default for convolutions on Ampere and would
silently cost ~3 mantissa bits, which is far outside this problem's 5e-5 tolerance --
the reference would then be wrong in the same direction a cheating kernel would be.
"""

import numpy as np
import torch
import torch.nn.functional as F

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


def prepare_input(scalars, buffers):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    N, C = int(scalars["BATCHES"]), int(scalars["CHANNELS"])
    IH, IW = int(scalars["IN_H"]), int(scalars["IN_W"])
    KH, KW = int(scalars["KERNEL_H"]), int(scalars["KERNEL_W"])

    x = torch.from_numpy(np.ascontiguousarray(buffers["input"], dtype=np.float32))
    w = torch.from_numpy(np.ascontiguousarray(buffers["filter"], dtype=np.float32))
    return {
        "x": x.to(dev).view(N, C, IH, IW),
        "w": w.to(dev).view(C, 1, KH, KW),   # depthwise: one filter per channel
        "stride": int(scalars["STRIDE"]),
        "padding": int(scalars["PADDING"]),
        "dilation": int(scalars["DILATION"]),
        "groups": C,
    }


def conv2d_dw_reference(prepared):
    out = F.conv2d(
        prepared["x"],
        prepared["w"],
        bias=None,
        stride=prepared["stride"],
        padding=prepared["padding"],
        dilation=prepared["dilation"],
        groups=prepared["groups"],
    )
    return out.reshape(-1)
