# Benchmark Landscape

Which public kernel benchmarks are worth pointing cacao at, what hardware each needs,
and what cacao is missing to run them. Measurements in this document were taken on the
project's RTX 3090 on 2026-08-06 unless stated otherwise.

---

## Scope

cacao is an automatic CUDA kernel optimizer, so "does the generated kernel beat a strong
baseline" is the right evaluation question and the KernelBench lineage is the relevant
literature. This document is about picking problems where that question has a meaningful
answer, and about not being fooled by the answer once you get it.

One nuance shapes *which* problems, not whether the frame applies. cacao's differentiator
over a single-shot LLM call is the KTT parameter-space search — ~100 s per iteration over
a `#define` space — plus multi-strategy branching. A problem with no tuning space gets
solved correctly by the first call, so it measures fusion quality, not search quality.
Prefer problems with a real parameter space when the goal is to show what cacao adds;
use the no-tuning-space ones as external yardsticks.

Two caveats on the external suites, both about fit rather than relevance:

- **kernelbench.com's `hard`/`cuda`/`mega` decks grade a differently-shaped agent** — one
  that clones repos, runs `ncu`, and emits arbitrary Triton/CUDA under no wall-clock
  budget. cacao's agent fills regions of a fixed KTT driver. Its `v3/v4_rtx3090` set is
  the part that fits.
- **Everything in `eval_results.csv` is internally comparable only.** No outside reader
  can check "1052 µs on kimi vs 328 µs on qwen." §6.2 supplies the one number they can.

## TL;DR

- **`level1/88_MinGPTNewGelu` is a calibration anchor, not a flagship.**
  KernelBench-Verified measured 8.52× on it, adversarially verified across four input
  distributions; this repo's independent 3090 headroom bound is 8.7×. Two methods,
  different hardware, agree to 2%. Use it to answer "is our agent leaving anything on the
  table" — but note it has nearly no tuning space, so it tests fusion, not autotuning
  (§6.2).
- **For an external problem that actually exercises autotuning, prefer
  `19_Mamba_Selective_Scan`** (§2): real tuning axes (chunk length over L, scan algorithm,
  state in registers vs shared memory, two-pass split) and no vendor library in the way.
- **The 3090 is not the bottleneck.** Every unwinnable result measured so far was
  unwinnable because of *problem selection*, not missing hardware. Fix selection first.
  Of 195 KernelBench L1+L2 tasks measured here, **42% are vendor-library-bound and 25%
  are already a single fused torch kernel** — two thirds are unwinnable before anyone
  writes a line of CUDA.
- **Two independent ways a task is unwinnable:** `lib_frac` (share of eager time inside
  cuBLAS/cuDNN) and `roofline_frac` (how close eager already is to the memory roofline).
  Together they give a numeric ceiling, `1 / max(lib_frac, roofline_frac)`, which is what
  makes the sweep actionable rather than just a filter.
- **The blocking gap for external benchmarks is the I/O boundary dtype.** cacao's
  buffers are `float` or integer C types only. Anything whose *contract* is bf16-in /
  bf16-out cannot be expressed, which rules out most of kernelbench.com.
- **Best card to buy if we buy one: A100** — but only after problem selection is fixed.

---

## 1. The two failure modes, measured

`scripts/headroom.py` (see §5) reports both per task.

### Library-bound (`lib_frac`)

Share of eager CUDA time inside cuBLAS / CUTLASS / cuDNN. High means you must beat a
vendor library head-on, which nothing in this project has managed.

Reference point — `problems/gemm_multiply_leakyrelu` (KernelBench L2/12), re-measured
this session at steady-state clocks:

| lane | cacao best | best torch path | verdict |
|---|---|---|---|
| fp32 | 6589 µs | 6236 min / 7071 med | tie, inside ±13% throttle noise |
| tf32 | 4004 µs | 4148–4207 µs | +5%, i.e. exactly the fused epilogue |
| fp16 | 1985 µs | 2244 µs | +13%, via drained fp16 accumulate |

`lib_frac` = 0.977. The fusable epilogue is ~165 µs of a 6.5 ms step — **2.5%** — and
that is the entire prize. The tf32 kernel's whole win *is* that 2.5%; its GEMM is at
parity with the one torch calls and ~6% behind cuBLAS 12.0 called directly.

### Bandwidth-saturated (`roofline_frac`)

`(bytes_in + params + bytes_out) / BW ÷ measured_eager_time`. Near 1.0 means eager is
already moving data at line rate and there is nothing to fuse away.

This is the one that is easy to miss. KernelBench resized its tasks, so:

| task | n | min traffic | eager | roofline_frac |
|---|---|---|---|---|
| L1/19 ReLU | 1 | 12.885 GB | 15212 µs | **1.039** |
| L1/20 LeakyReLU | 1 | 12.885 GB | 15215 µs | **1.039** |
| L1/21 Sigmoid | 1 | 12.885 GB | 15229 µs | **1.038** |

(`roofline_frac` slightly over 1.0 means the 815 GB/s constant is ~4% conservative;
achieved is ~847 GB/s. Spec is 936 GB/s.)

L1/19 ReLU is precisely the task where KernelBench-Verified found a **374× claimed
speedup that was a shape-conditional identity shortcut**. That is not a coincidence:
no honest kernel can beat a saturated one, so any large number on a `roofline_frac ≈ 1`
task is a bug or a cheat by construction.

**Honest accounting of what this metric adds as a *filter*:** not much. These three are
`n_kernels == 1`, so the older screen (`lib_frac` low AND `n_kernels >= 2`) already
excluded them. Across all 195 tasks, exactly **one** passes the old screen while being
saturated (`level1/4_Matrix_vector_multiplication_`, n=2, `roofline_frac` 0.864).

Its real value is **quantitative**: it converts a binary screen into a numeric ceiling,
`1 / max(lib_frac, roofline_frac)`. That is what produced the 8.7× bound on L1/88 which
matches the published 8.52× (§6.2) — a pass/fail screen could never have told us that.

---

## 2. Suites surveyed

| suite | tasks | target HW | scoring | runs on 3090? | cacao fit |
|---|---|---|---|---|---|
| [KernelBench](https://github.com/ScalingIntelligence/KernelBench) L1/L2 | 100 + 100 | any | speedup vs eager | **yes** | **good** — plain torch modules |
| KernelBench L3/L4 | full models | any | speedup vs eager | mostly | poor — multi-op, not one kernel |
| [KernelBench-Verified](https://github.com/facebookresearch/kernel_bench_verified) | same 250 + hidden tests | H200 in paper | speedup, verified | **yes** | **use its test suite regardless** |
| [KernelBench-X](https://arxiv.org/html/2605.04956v1) | KernelBench across 6 GPUs | A100/H20/H800/L20/4090/5090 | speedup vs eager | n/a (results only) | reference data |
| [kernelbench.com](https://github.com/Infatoshi/kernelbench.com) `hard` | 6 | H100 / RTX PRO 6000 / B200 | % of roofline | partly | blocked on dtype |
| kernelbench.com `cuda` | 4 | RTX PRO 6000 | % of roofline | no | out of scope — model-block scale |
| kernelbench.com `mega` | 1 | H100 | speedup | no | out of scope — whole model block |
| kernelbench.com `v3/v4_rtx3090` | **3** | **RTX 3090** | speedup | **yes** | **best external fit** |
| [MultiKernelBench](https://github.com/wzzll123/MultiKernelBench) | **285** in 14 categories | NVIDIA GPU / Huawei NPU / Google TPU | speedup vs eager | **yes** | **good** — KernelBench-shaped, finer categories |
| [TritonBench](https://arxiv.org/pdf/2603.19173) G / T | 2 splits | NVIDIA | speedup vs reference | yes | poor — Triton DSL, cacao writes CUDA C++ |
| [SOL-ExecBench](https://github.com/NVIDIA/SOL-ExecBench) | **235** from 124 production models | **Blackwell** (B200) | SOL Score vs analytic bound | partly (BF16 subset) | methodology worth stealing |
| FlashInfer-Bench | 26 inference primitives | NVIDIA | speedup | likely | traced from vLLM/SGLang production shapes |

Still not examined: FastKernels, KernelSkill, EGG, AdaExplore, KForge, Kevin, CuGEdit,
Kernel-Smith, METR's study. Several of these are *systems* rather than benchmarks — they
report on KernelBench — so they are sources of comparison numbers, not new task sets.

#### The two worth knowing about

**MultiKernelBench** (Nanjing University, arXiv 2507.17773) — 285 tasks across 14
kernel categories on three backends behind a modular abstraction layer. It exists
specifically because KernelBench has "limited hardware support, coarse-grained kernel
categorization, and imbalanced task coverage." Same torch-module contract as KernelBench,
so the headroom screen in §1 applies unchanged and it runs on this card. The obvious next
sweep after §6.

**SOL-ExecBench** (NVIDIA, arXiv 2603.19173) — 235 CUDA problems extracted from 124
production models across language, diffusion, vision, audio and video, scored not by
speedup but by how much of the gap between a baseline and an **analytically derived
Speed-of-Light bound** the kernel closes. Its framing is a direct argument against the
metric cacao currently uses:

> progress is constrained by benchmarks that reward speedup over software baselines
> rather than proximity to hardware-efficient execution

Hardware puts it out of reach — BF16/FP8/NVFP4 targeting Blackwell, with kernels
"expected to rely on Blackwell-specific capabilities." But **its harness is worth copying
regardless**: GPU clock locking, L2 cache clearing between runs, isolated subprocess
execution, and static-analysis checks against known reward-hacking strategies. Clock
locking in particular is a measured problem here — cold-clock baselines inflated this
session's numbers by 7–8% (§5), and cacao clears no cache between tuning configurations.

Corroboration from a third direction: Sakana's *Towards Robust Agentic CUDA Kernel
Benchmarking* reports **~40 KernelBench v0 tasks are problematic** — inefficient eager
baselines and insufficient output variation across seeds. That is an independent estimate
of the same rot §6 quantifies from measurement.

### 2.1 Autotuning suites — adjacent family, not the primary target

Everything above evaluates *kernel generation by speedup*, which is cacao's frame. There
is a second, older family built to benchmark **autotuners** rather than generators. It
does not measure generation and has no LLM in the loop, so it is not the primary target —
but it is the closest match to cacao's *mechanism* (a designed `#define` space searched by
KTT), and it supplies expert-written reference points that no generation benchmark does.

#### The KTT benchmark set — same toolkit, same institution

[Petrovič et al., *A Benchmark Set of Highly-efficient CUDA and OpenCL Kernels and its
Dynamic Autotuning with Kernel Tuning Toolkit*](https://arxiv.org/abs/1910.08498) (FGCS
2020), from the Institute of Computer Science, Masaryk University Brno — the KTT authors.
Ten autotunable kernels spanning image processing, linear algebra, computational
chemistry, and PDE solvers. They ship with KTT under `Examples/`.

| kernel | tuning dimensions | configurations |
|---|---|---|
| GEMM | 15 | 241,600 |
| Transpose | 9 | 10,752 |
| N-body | 8 | 9,408 |
| Convolution (2D) | 10 | 5,248 |
| BiCG | 11 | 5,122 |
| Coulomb 3D | 8 | 1,260 |
| Hotspot | 6 | 480 |
| GEMM batched | 11 | 424 |
| 3D Fourier Reconstruction | 6 | 360 |
| Reduction | 5 | 175 |

Their metric is **roofline efficiency** — `max(MEMops/time / MEMpeak, ALUops/time /
ALUpeak)`, counting operations *essential to the task*, not operations the algorithm
happens to execute. That is the same quantity as `roofline_frac` in §1 and the same
quantity kernelbench.com grades on.

Published results, on consumer NVIDIA cards one and two generations from ours:

| kernel | RTX 2080 Ti | GTX 1070 |
|---|---|---|
| Coulomb 3D | 91.8% | 91.4% |
| N-body | 89.7% | 86.6% |
| BiCG | 88.3% | 84.7% |
| Transpose | 87.1% | 80.2% |
| GEMM batched | 86.8% | 81.4% |
| GEMM | 79.8% | 80.6% |
| Reduction | 68.7% | — |
| Hotspot | 1.35× over Rodinia | — |

**Why this is the best available evaluation for cacao:**

1. **Same toolkit.** These tuning spaces are KTT `AddParameter` spaces. No porting layer.
2. **Same metric**, and one that does not depend on how naive someone made a baseline.
3. **The right baseline.** The comparison is *expert-written kernel + KTT autotuning*
   versus *LLM-written kernel + KTT autotuning*. That isolates exactly what cacao adds,
   which "did you beat eager PyTorch" never does.
4. **Consumer-GPU numbers already published**, so a 3090 result is interpretable.
5. **cacao already contains one.** `problems/covariance` corresponds to KTT's
   `Examples/Covariance`; `Examples/` also holds Bicg, Nbody, Reduction, Transpose,
   ClTuneGemm, Convolution3d, CoulombSum2d/3d, FluidSimulation, Sort.

The bar is concrete and hard: reach ~88% of roofline on BiCG, ~90% on N-body, ~80% on
GEMM. Falling short is informative in a way that beating a decomposed GELU is not.

#### BAT

[NTNU-HPC-Lab/BAT](https://github.com/NTNU-HPC-Lab/BAT), a GPU benchmark suite built to
*compare autotuners*, drawn from SHOC, Rodinia, KTT, and the Netherlands eScience Center:
GEMM, Nbody, DeDisp, ExpDist, PnPoly, Convolution, Hotspot, MD5Hash, TRIAD. BAT 2.0's
listed integrations are Kernel Tuner, OpenTuner, Optuna, SMAC3, and its own Mintuner —
KTT was supported in the original paper but is not in the current compatibility matrix,
so a KTT path may need reviving. Same family, one step further from cacao than the KTT
set itself.

### What the literature says actually wins

Two independent studies converge on the same characterization.

**KernelBench-Verified** (Meta, H200) — under a corrected protocol **no model beats
PyTorch**: best (GPT-5.5) drops from 1.43× to **0.88× geomean**. ~35% of claimed
speedups were eliminated:

- 21.2% by TF32 baseline correction
- 11.6% by a hidden 4-distribution test suite (×3.0, ×0.01, negated)
- 2.4% by both

Named genuine survivor: **Problem 88 (GELU), 8.52×**, passes all four distributions.
Named fake: Problem 19 (ReLU), 374×, identity shortcut.

**KernelBench-X** (six GPUs) — pooled median speedup **1.0008×**, 46.6% of *correct*
kernels slower than eager. Slower-than-torch fraction ranges **18% on A100 to 76% on
L20**. Named genuine: the `logit` task at **~3.35× on RTX 4090**, reproduced
independently by GEAK, Claude, and AutoTriton.

Both describe the winning class the same way — KernelBench-X calls it **"local,
single-path semantics"**: each output element depends on exactly one input element.
Decomposed elementwise chains where torch launches N passes and a fused kernel does one.

**That win is structural, not architectural.** It comes from collapsing N bandwidth
passes into 1, which holds on any GPU. That is why the same shape of task wins on both
an H200 and a 4090, and why it transfers to a 3090. Anything GEMM- or conv-rooted is a
precision/architecture game instead, and that is where this project has consistently lost.

Note: KernelBench-Verified's #1 correction — TF32 baselines — is exactly the error that
made this repo's tf32 branch read 1.57× when it is really 1.05×. The correction is
load-bearing, not pedantry.

### kernelbench.com's three RTX 3090 problems

`benchmarks/v3/problems/v4_rtx3090/`, the only public problem set naming our card:

| problem | reference | shape | why it's winnable | blocker |
|---|---|---|---|---|
| **19_Mamba_Selective_Scan** | Python loop over **4096 timesteps** | B=2 L=4096 D=256 N=16 | ~20k launches, zero cuBLAS; ~515 µs bandwidth floor in fp32 | **none — fp32 is a listed supported precision** |
| **28_Punica_SGMV** | Python loop over 8 rows | N=8, 8192→8192, 4 adapters, r=16 | 16 tiny GEMM launches vs one fused kernel; 5 MB total | bf16 contract |
| **18_POD_Attention** | Python loop over 8 requests | 4 prefill qL=512 + 4 decode, H=32, GQA 4, D=128 | materializes a 32×512×2048 score matrix; ~970 µs compute floor | bf16 contract |

Mamba is the standout: no dtype blocker, the tuning knobs (chunk length over L,
warp-vs-block scan, N=16 in registers) are exactly the `#define` space KTT searches, and
the two-pass scan structure fits the multi-launch launcher region the fp16 GEMM branch
already uses.

Caveat that applies to all three: the references are *deliberately naive Python loops*.
Speedup-vs-reference would report a meaningless ~1000×. See §4 on the `baseline:` gap.

---

## 3. Hardware

| card | arch | mem | BW | adds over 3090 | unlocks |
|---|---|---|---|---|---|
| **RTX 3090** (have) | GA102 sm_86 | 24 GB | 936 GB/s spec, **847 measured** | — | — |
| RTX 4090 | AD102 sm_89 | 24 GB | 1008 GB/s | **fp8** (e4m3/e5m2) | kernelbench.com FP8 GEMM |
| L40S | AD102 sm_89 | 48 GB | 864 GB/s | fp8, 2× memory | larger shapes |
| A100 | GA100 sm_80 | 40/80 GB | 1.55–1.94 TB/s | HBM, no fp8 | **best measured win rate** |
| H100 | GH100 sm_90 | 80 GB | 3.35 TB/s | fp8, **TMA, wgmma, clusters** | kernelbench.com `hard`/`mega`, literature parity |
| B200 | sm_100 | 192 GB | 8 TB/s | fp4, 2nd-gen TMA | kernelbench.com B200 deck |

**What the 3090 has:** fp32 (35.6 TFLOP/s), tf32 (35.6), bf16/fp16 with fp32 accumulate
(71.2), fp16 with **fp16** accumulate (142.4 — the GeForce 2× path the gemm branch
exploits), int8 (~284 TOPS), int4 (~568 TOPS), `cp.async`.

**What it lacks:** fp8, fp4, TMA, `wgmma`, thread-block clusters, distributed shared
memory. Everything Hopper-specific in CUTLASS is off the table.

### Which card to test on

- **For comparability with published results:** H100/H200. That is where
  KernelBench-Verified and kernelbench.com's `hard` and `mega` decks live. Nothing we
  produce on a 3090 is directly comparable to any published number.
- **For the best odds of a genuine win:** **A100**. KernelBench-X measured only 18% of
  correct kernels slower than torch there — the best of six cards, versus 76% on L20.
  Its bandwidth-to-compute ratio favours exactly the bandwidth-bound fusions that are the
  only reliably winnable class.
- **Cheapest capability unlock:** RTX 4090 — same 24 GB, ~20% more bandwidth, adds fp8.
- **Do not buy yet.** Nothing measured so far was lost to missing hardware. L2/12 was lost
  to `lib_frac` 0.977; L1/19 is unwinnable at `roofline_frac` 1.04 on *any* card.

---

## 4. What cacao is missing

Ordered by how much it blocks.

### 4.1 I/O boundary is fp32/int only — **blocking**

`utils/inputs.py:44` `_literal()` returns `f"{float(value)}f"` for `"float"` and
`str(int(value))` for everything else; buffer generation picks
`uniform_real_distribution` only for `float`, `uniform_int_distribution` otherwise. So
`__half`, `__nv_bfloat16`, and even `double` do not compile. KTT's
`SideBySideComparison` (`utils/framework.py:133`) takes a `double` tolerance and has no
half type, so a half-precision *output* buffer cannot be validated either.

**Consequence: reduced precision must live entirely inside the kernel.**
`gemm_multiply_leakyrelu` already works this way — fp32 buffers, an fp16 conversion
prologue costing 541 µs of its 1985 µs total.

That is fine for GEMM-shaped work. It blocks anything whose *contract* is bf16-in /
bf16-out: kernelbench.com's `hard` and `cuda` decks, and 2 of the 3 `v4_rtx3090`
problems. Escape hatch is `init: custom` with `unsigned short` storage plus manual bit
patterns, but validation of the output still has no half path.

### 4.2 No hidden-test / multi-distribution validation — **blocking for credibility**

cacao validates against exactly one input distribution, the one in `inputs.yaml`.
KernelBench-Verified's headline result is that **11.6% of claimed wins die on a second
distribution**. `rules.forbid` catches *source patterns*; it cannot catch a behavioural
shortcut like returning the input unchanged when a shape condition holds.

Until this exists, any speedup this project reports is unverified in the specific sense
the literature now considers mandatory. The `hidden_tests/` directory in
`facebookresearch/kernel_bench_verified` is directly reusable.

### 4.3 No roofline denominator

Speedup is vs the reference's wall time. For any problem whose reference is a naive loop
(all three `v4_rtx3090` ones) that number is meaningless, and it is not merely cosmetic:
it feeds `prompts/decide.py:47` ("implemented correctly but gave no speedup → do not
retry"), so branches stop believing they have won.

`save_reference_time` is `O_CREAT|O_EXCL` first-write-wins (`utils/results.py:213`) and
`engine/master.py:63` returns early when the file exists, so pre-writing
`output/reference_time.json` already works as a manual override. The durable fix is a
`baseline:` block in `problem.yaml` (`kind: roofline|sota|reference`, `time_us`, `note`)
that seeds the file and renders `note` into the decide/propose prompts.

### 4.4 One shape per problem

`inputs.yaml` describes a single shape. KernelBench-X and kernelbench.com both sweep 5
shapes per task, and a kernel tuned to one shape routinely regresses on another. Today
this needs one problem directory per shape.

### 4.5 Rules reach only the authoring prompt

`rules.text` is rendered by `nodes/author.py:220` into `prompts/author.py:82-83` and
nowhere else. Decide, propose, plan, and strategize never see it — despite the module
docstring claiming rules appear in every prompt that "writes **or judges**" a kernel.
Rules are also absent from the non-agentic fallback path entirely (`nodes/implement.py`
and `nodes/configure.py` do not load them, and `forbid` is enforced only in
`agentic/tools.py:295`).

### 4.6 The frontend silently destroys `rules:`

`CreateProblemRequest` (`api/schemas.py:19`) has no `rules` field, and `update_problem`
(`api/problems.py:284`) builds the yaml dict from scratch with no read-merge before
`_write_problem_files` opens `problem.yaml` with `"w"`. **Any hand-added key — `rules:`
included — is deleted the first time a problem is saved through the UI.** This is already
true today for the fp32-accumulation rules the module was written for.

---

## 5. Recommended next steps

1. **Land `scripts/headroom.py`** (currently in a scratchpad) as a pre-flight check when
   adding a problem. It reproduces known ground truth: L2/12 at `lib_frac` 0.977 and
   `eager_us` 6284 against the 6280 in `reference_time.json`.
2. **Port `level1/88_MinGPTNewGelu`** (§6.2) — the one task with an externally verified
   speedup our own measurement independently corroborates. If cacao cannot reach ~8×
   there, the gap is in the agent, not the problem set. It has almost no tuning space, so
   treat the result as a yardstick rather than a demonstration of what cacao adds.
3. **Then work down §6.1**, which needs no vendor library beaten. Avoid §6.3 until §6.1
   produces a win. The existing student / mmul / flood problems stay useful as the
   parameter-space-rich end of the corpus; these add external comparability.
4. **Adopt the measurement hygiene** before believing any result. Three from
   KernelBench-Verified — warm-clock baseline (cold-clock measurement inflated this
   session's numbers 7–8%), TF32-matched baseline wherever a matmul is present, and the
   4-distribution hidden tests. Three more from NVIDIA's SOL-ExecBench harness — locked
   GPU clocks, L2 cache clearing between runs, and static-analysis checks for known
   reward-hacking patterns (cacao's `rules.forbid` is the natural home for the last one).
5. **Sweep MultiKernelBench** with `headroom.py` once §6 is acted on — 285 tasks in 14
   categories, same torch-module contract, runs on this card, and built specifically to
   fix KernelBench's imbalanced coverage.
6. **Port `19_Mamba_Selective_Scan`** as the first kernelbench.com problem — no dtype
   blocker, and the tuning space is the whole problem.
7. Fix §4.6 before anyone edits a problem with rules through the UI.

---

## 6. Full L1+L2 sweep (RTX 3090, 2026-08-06)

200 tasks, 195 measured, 5 failed — all OOM on this 24 GB card (`22_Tanh`, `25_Swish`,
`30_Softsign` want 6 GB intermediates; `38_L1Norm_`, `39_L2Norm_` want 8 GB).

How the 195 break down:

| category | count | share |
|---|---|---|
| `lib_frac > 0.9` — vendor-library-bound, the L2/12 trap | 81 | 42% |
| `n_kernels == 1` — torch already ships one fused kernel | 49 | 25% |
| `roofline_frac > 0.8` — already saturating DRAM | 22 | 11% |

**Headroom bound = `1 / max(lib_frac, roofline_frac)`.** You cannot go below the vendor
library's time, and you cannot go below the memory floor; whichever binds first is the
ceiling. Note this corrects a naive reading of `roofline_frac` alone — a conv task can
show `roofline_frac` 0.007 and still be capped at 9.6× because it is compute-bound.

### 6.1 No library to match — the safe targets

`lib_frac < 0.02`, so a fused kernel competes only against torch's own launches. Nothing
here requires beating cuBLAS or cuDNN.

| task | n | eager µs | min traffic | roof_frac | bound |
|---|---|---|---|---|---|
| **level1/88_MinGPTNewGelu** | 6 | 5720 | 0.54 GB | 0.115 | **8.7x** |
| level1/100_HingeLoss | 4 | 35146 | 4.29 GB | 0.150 | **6.7x** |
| level1/98_KLDivLoss | 6 | 13869 | 2.15 GB | 0.190 | **5.3x** |
| level1/99_TripletMarginLoss | 5 | 15078 | 3.22 GB | 0.262 | **3.8x** |
| level1/92_cumsum_exclusive | 4 | 37610 | 8.59 GB | 0.280 | **3.6x** |
| level1/91_cumsum_reverse | 2 | 33000 | 8.59 GB | 0.319 | **3.1x** |
| level1/94_MSELoss | 3 | 30093 | 8.59 GB | 0.350 | **2.9x** |
| level1/36_RMSNorm_ | 5 | 46195 | 15.03 GB | 0.399 | **2.5x** |
| level1/93_masked_cumsum | 2 | 24155 | 9.66 GB | 0.491 | **2.0x** |
| level1/95_CrossEntropyLoss | 2 | 1325 | 0.54 GB | 0.497 | **2.0x** |
| level1/96_HuberLoss | 2 | 19981 | 8.59 GB | 0.527 | **1.9x** |

### 6.2 The cross-validation

**`level1/88_MinGPTNewGelu` is KernelBench-Verified's Problem 88.** It is the decomposed
minGPT GELU:

```python
0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
```

on an (8192, 8192) fp32 input — 6 elementwise kernels, 268 MB in, 268 MB out, zero
library time.

| source | figure |
|---|---|
| KernelBench-Verified, H200, verified genuine, survives all 4 hidden-test distributions | **8.52×** |
| This sweep's independently computed 3090 bound (`1 / roofline_frac`) | **8.7×** |

Two independent methods on different hardware agree to within 2%. This is the strongest
external evidence available that a specific speedup is real and reachable here — the
published number is a *measured, adversarially verified* result, and our bound says the
same headroom exists on this card. It is also the exact task class both papers identify
as genuinely winnable ("local, single-path semantics").

Everything cacao needs is already supported: fp32 boundary, single fused kernel, one
shape, elementwise-only. No dtype blocker, no roofline-denominator problem (torch's 5720
µs is a real baseline, not a naive loop).

### 6.3 Higher ceilings, but you must match cuDNN first

`0.02 ≤ lib_frac < 0.35` with ≥4 kernels — mostly-fusable conv chains. The bound is
higher, but realizing it means re-implementing the convolution at least as fast as cuDNN,
which is the same trap L2/12 sprung. Treat as stretch goals, not starting points.

| task | n | eager µs | lib_frac | bound |
|---|---|---|---|---|
| level2/92_Conv2d_GroupNorm_Tanh_HardSwish_ResidualAdd_LogSumExp | 16 | 12544 | 0.104 | **9.6x** |
| level2/57_Conv2d_ReLU_HardSwish | 7 | 9292 | 0.138 | **7.2x** |
| level2/21_Conv2d_Add_Scale_Sigmoid_GroupNorm | 8 | 16337 | 0.153 | **6.5x** |
| level2/82_Conv2d_Tanh_Scaling_BiasAdd_Max | 6 | 28180 | 0.193 | **5.2x** |
| level2/85_Conv2d_GroupNorm_Scale_MaxPool_Clamp | 9 | 6465 | 0.205 | **4.9x** |
| level2/87_Conv2d_Subtract_Subtract_Mish | 5 | 25320 | 0.213 | **4.7x** |
| level2/69_Conv2d_HardSwish_ReLU | 5 | 4993 | 0.260 | **3.8x** |
| level2/71_Conv2d_Divide_LeakyReLU | 5 | 5005 | 0.260 | **3.8x** |

Raw data: `headroom.json` (per-task `lib_frac`, `roofline_frac`, `n_kernels`, `eager_us`,
`gb_min`, and the three hottest kernel names).
