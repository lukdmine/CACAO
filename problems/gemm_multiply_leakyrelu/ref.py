"""Python reference for KernelBench level2/12_Gemm_Multiply_LeakyReLU.

The oracle is the KernelBench nn.Module itself, run on the GPU. Its weights are NOT
initialised by torch: they are loaded from the buffers the C++ driver filled and dumped
to cacao_in_gemm_weight.bin / cacao_in_gemm_bias.bin. So the reference and the tuned
kernel are guaranteed to see bit-identical weights — there is no seed to keep in sync.

The class is named `_Model`, not `Model`, on purpose. utils.python_ref_runner picks the
reference by scanning dir(module) alphabetically for the first public callable defined
here; a class is callable and 'Model' sorts before any lowercase function name, so a
public class would be picked instead of the function and called as Model(scalars,
buffers). The leading underscore makes the loader skip it.

Layout is torch's: ROW-MAJOR throughout.
  x            (kBatch, kIn)   flat index row*kIn + col
  gemm_weight  (kOut,   kIn)   flat index o*kIn + i     -- nn.Linear stores (out, in)
  gemm_bias    (kOut,)
  out          (kBatch, kOut)  flat index row*kOut + col
"""

import torch
import torch.nn as nn


class _Model(nn.Module):
    """KernelBench level2/12, verbatim apart from the leading underscore."""

    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super(_Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.multiplier = multiplier
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, x):
        x = self.gemm(x)
        x = x * self.multiplier
        x = self.leaky_relu(x)
        return x


def prepare_input(scalars, buffers):
    """Everything that is not the computation: model construction and the H2D copies.

    Split out because utils.python_ref_runner times only the reference callable, and
    the recorded time becomes the denominator of every speedup this problem reports.
    Building an 8192x8192 nn.Linear and copying 256 MB of weights to the device is not
    work the tuned kernel does — the kernel is handed device-resident buffers — so
    including it would overstate the reference and flatter every kernel measured
    against it.
    """
    batch = scalars["kBatch"]
    in_features = scalars["kIn"]
    out_features = scalars["kOut"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _Model(
        in_features,
        out_features,
        scalars["multiplier"],
        scalars["negative_slope"],
    ).to(device)

    # .copy() because the runner hands out np.frombuffer views, which are read-only —
    # torch.from_numpy warns on those and refuses to share the storage.
    with torch.no_grad():
        model.gemm.weight.copy_(
            torch.from_numpy(buffers["gemm_weight"].copy()).view(out_features, in_features)
        )
        model.gemm.bias.copy_(torch.from_numpy(buffers["gemm_bias"].copy()))
        x = torch.from_numpy(buffers["x"].copy()).to(device).view(batch, in_features)

    return {"model": model, "x": x}


def gemm_multiply_leakyrelu_reference(prepared):
    """The timed region: one forward pass over device-resident inputs.

    This is what the tuned kernel is competing against, so it is all that is measured.
    The runner warms it once and averages a few repeats.
    """
    with torch.no_grad():
        return prepared["model"](prepared["x"])
