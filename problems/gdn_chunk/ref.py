"""Python reference for one Gated DeltaNet chunked forward pass at Qwen3.8-27B dims.

The oracle is `torch_chunk_gated_delta_rule`, vendored verbatim (modulo the leading
underscore and dropped kwargs) from transformers/models/qwen3_5/modeling_qwen3_5.py --
the Qwen authors' own reference for this exact model family. It is vendored rather than
imported so the oracle is pinned: a benchmark whose definition of "correct" drifts with
the installed transformers version is not a benchmark. `tests/` re-checks it against the
installed transformers and against fla's Triton kernel.

Everything in the oracle runs in fp32 (`.to(torch.float32)` on entry), which is why this
problem's I/O boundary is fp32 end to end and why `mamba_ssm_dtype: float32` in the real
config is not a simplification we imposed.

LAYOUT -- all buffers are row-major, token-major, matching what the model produces and
what fla takes as [B, T, H, D]:

  q, k    (kT, kHk, kD)   flat  t*kHk*kD + h*kD + d      16 QK heads
  v, o    (kT, kHv, kD)   flat  t*kHv*kD + h*kD + d      48 V  heads
  g, beta (kT, kHv)       flat  t*kHv + h
  state   (kHv, kD, kD)   flat  h*kD*kD + dk*kD + dv

GQA is real: 16 QK heads feed 48 V heads, so V head j reads QK head j // 3. The expansion
is `repeat_interleave(3)`, matching the model, so the mapping is floor-division and NOT
modulo. A kernel that uses `j % 16` computes a different problem and will fail validation.
"""

import numpy as np
import torch
import torch.nn.functional as F


def _l2norm(x, dim=-1, eps=1e-6):
    """Aligned with fla's l2norm, per the transformers docstring."""
    return x / torch.clamp(x.norm(2, dim=dim, keepdim=True), min=eps)


def _torch_chunk_gated_delta_rule(
    query, key, value, g, beta, chunk_size=64,
    initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=False,
):
    """Vendored from transformers.models.qwen3_5.modeling_qwen3_5."""
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim,
                    dtype=value.dtype, device=value.device)
        if initial_state is None else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def prepare_input(scalars, buffers):
    """Untimed: host->device copies, reshaping, and the GQA expansion.

    Split out because utils.python_ref_runner times only the reference callable. None of
    this is work the tuned kernel does -- it is handed device-resident buffers -- so
    including it would flatter every kernel measured against it. (This problem overrides
    output/reference_time.json anyway; see problem.yaml.)
    """
    T, Hk, Hv, D = scalars["kT"], scalars["kHk"], scalars["kHv"], scalars["kD"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def t(name, shape):
        # .copy() because the runner hands out read-only np.frombuffer views.
        return torch.from_numpy(buffers[name].copy()).to(dev).view(*shape)

    q = t("q", (1, T, Hk, D))
    k = t("k", (1, T, Hk, D))
    v = t("v", (1, T, Hv, D))
    g = t("g", (1, T, Hv))
    beta = t("beta", (1, T, Hv))

    # GQA: 16 QK heads -> 48 V heads. repeat_interleave, so V head j reads QK head j//3.
    rep = Hv // Hk
    q = q.repeat_interleave(rep, dim=2)
    k = k.repeat_interleave(rep, dim=2)
    return {"q": q, "k": k, "v": v, "g": g, "beta": beta}


def gdn_chunk_reference(prepared):
    """The timed region: one chunked GDN forward over device-resident inputs.

    Returns `o` only. The runner validates one target buffer per invocation and hands the
    callable no way to tell which one was asked for, so `state` is declared
    `validate: false` in inputs.yaml. That is sufficient rather than lax: o[chunk i]
    consumes the recurrent state left by chunk i-1, so any error in the state recurrence
    corrupts every subsequent chunk of o. The only thing left uncovered is the final
    state write after the last chunk, which is why problem.yaml carries a text rule
    requiring it.
    """
    with torch.no_grad():
        o, _ = _torch_chunk_gated_delta_rule(
            prepared["q"], prepared["k"], prepared["v"], prepared["g"], prepared["beta"],
            chunk_size=64, output_final_state=True, use_qk_l2norm_in_kernel=True,
        )
    return o.reshape(-1)
