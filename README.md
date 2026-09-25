# INT8 convolution kernel lab

This is a CUDA-only experiment harness for signed INT8 implicit-GEMM
cross-correlation with an INT32 bias and output:

```text
input:  NHWC int8
filter: KRSC int8
bias:   K int32
output: NHWK int32
```

It deliberately has no MATLAB, MEX, CGIR, or cuDNN dependency.  Its purpose is
to measure a dispatch policy, not to replace a production convolution library.

Three kernel families have the same input/output contract:

- `generic`: staged implicit GEMM, covering every shape with `C % 16 == 0` and
  the fallback for anything the specialized candidates decline.  Its CTA tile,
  warp tile and staging depth are template parameters
  (`GenericTile<CtaM, CtaN, WarpM, WarpN, StageK>`, overridable with
  `INT8_LAB_GENERIC_*`), and the kernels derive their fragment counts, loop
  bounds and accumulator extents from them.  294 combinations were checked
  exhaustively against the CPU oracle on two shapes and both MMA families.
- `spatial_strip`: a `128 x 64 x 64` specialization.  It stages one
  horizontally contiguous `130 x 64` input strip and reuses it across the three
  filter columns of a 3x3, horizontal-stride-one convolution.
- `spatial_halo_splitk`: the halo tile with its channel-group reduction
  partitioned across `Splits` CTAs that atomically accumulate into one output
  tile.  It exists for shapes where `gemm_m x gemm_n` is too small to fill the
  GPU but `gemm_k` is deep -- a 7x7 output with `K=512` launches 16 CTAs onto
  36 SMs and then serialises a 4608-deep reduction inside each.  It costs a
  zeroing pass plus `Splits` atomics per output element, so it loses badly when
  the output is large; the driver measures both and keeps the better one.
- `spatial_halo`: the same reuse idea on both axes.  It stages one 2D input
  patch per channel group and reuses it across *all* `R*S` taps, so it moves
  less activation traffic than the strip and drops the strip's `out_w` divisor
  requirement -- output edges are predicated instead.  The patch aspect ratio
  is a compile-time knob (`INT8_LAB_HALO_TILE_H`/`_W`, default `4 x 32`).
  The output-channel tile (`cta_n`, `INT8_LAB_HALO_CTA_N`) is searched over 64
  and 128.  The A slab does not depend on it, so doubling it buys twice the MMA
  per A-stage for about 10% more shared memory -- but it also halves the grid,
  and that trade splits exactly on stride: every stride>1 shape gains
  (`big_3x3_s2` 1.22x, `resnet_s4_down` 1.14x) and every stride-1 3x3 loses
  (`darknet_res_52_expand` 0.81x, `dilated_3x3` 0.82x).
  Channels staged per pass (`stage_k`) is searched at run time over 64 and 32,
  because it is not a global default: 64 wins when the patch is compact, but
  stride or a large filter inflates the halo until the staged slab, not the
  register file, binds occupancy.  A `4 x 32` patch at stride 2 needs 41 KB and
  gets 2 CTAs/SM at 64, against 21 KB and 4 at 32 -- worth 1.55x on
  `big_3x3_s2`, and it makes `big_5x5_s2` eligible at all, while costing 10% on
  `big_3x3_s1`.

Unsupported shapes are reported as ineligible, rather than silently using a
different algorithm.

## Requirements

- NVIDIA CUDA Toolkit 12+ (`nvcc`)
- An NVIDIA GPU with compute capability 7.5 or newer
- Python 3 for specialization and sweep scripts

No GPU is required to compile.  A GPU is required to run or validate a kernel.

## One case

```bash
make CASE=smoke ARCH=sm_75
./build/conv_lab --strategy both
```

`MMA=auto` (the default) selects the same INT8 MMA family as CGIR for the
generic path from the compile target: `m8n8k16` for `sm_75`, and `m16n8k32`
for `sm_80+`. Override it only when intentionally inspecting a non-default
code path:

```bash
make CASE=smoke ARCH=sm_86 MMA=sm80
make CASE=big_3x3_s1 ARCH=sm_75 MMA=sm75
```

The executable checks the selected kernel(s) against a CPU INT32 oracle before
timing them.  It emits a CSV row suitable for a dispatch-tuning table:

```bash
./build/conv_lab --strategy both --csv
```

Use `--strategy generic`, `--strategy strip`, `--strategy halo`, or
`--strategy splitk` to isolate a candidate; `--strategy both` runs every
eligible one.  The CSV carries one eligibility flag and one timing column per
candidate (`eligible_strip`, `eligible_halo`, `eligible_splitk`, `generic_ms`,
`strip_ms`, `halo_ms`, `halo_splitk_ms`), the configuration that produced each
halo number (`halo_stage_k`, `halo_cta_n`, `halo_splits`,
`halo_splitk_stage_k`), and `winner`/`best_ms`.  The halo and split-K
candidates search twenty configurations per shape -- two `cta_n` by two
`stage_k` by five split counts -- so a build covers the whole space and the CSV
records which point won.  `--coverage` prints what
fraction of output channels, `ox`, and `oy` the correctness sampler reaches.

The oracle check is exhaustive below 65536 outputs.  Above that it walks a
golden-ratio stride coprime to the output size, plus a directed set of padded
borders, row ends, and channel tails.  A stride sharing a factor with `K` --
which a uniform `i * n / count` walk always has -- pins every sample to output
channel 0 and hides an entire class of epilogue bugs.

Both spatial kernels hold 64 INT32 accumulators per thread and need
`INT8_LAB_MIN_CTAS_PER_SM` (default 4 for `m16n8k32`, 3 for `m8n8k16`) to reach
usable occupancy; one higher spills and costs roughly 2x.  Re-tune it per
architecture.

## Sweep workloads

```bash
./scripts/sweep.py --cases smoke,resnet_s2_3x3,big_3x3_s1 \
    --output build/dispatch_measurements.csv
```

The script rebuilds one compile-time specialization per case, invokes both
eligible kernels, and writes one CSV row per workload.  Point it at another
CSV file with `--configs`.  Its required columns match `workloads.csv`.

`workloads.csv` contains 41 cases: ResNet stages, Darknet-53/YOLOv3,
batch-scaled residual layers, bottleneck and pointwise layers, dilated and
asymmetric filters, patch embedding, large stress cases, and non-multiple-of-8
output-channel tails.  The tail cases deliberately exercise the generic
fallback rather than the strip specialization.

## cuDNN comparison

```bash
./scripts/sweep.py --arch sm_120 --with-cudnn \
    --output build/dispatch_vs_cudnn.csv
```

The baseline uses cuDNN's graph (backend) API and measures two contracts per
shape, because they answer different questions:

- `cudnn_f32_ms`: `int8` NHWC convolution with INT32 accumulation into a
  `float` output.  Four output bytes per element, matching this lab's INT32,
  so write traffic is equal.  **This is the like-for-like kernel comparison.**
- `cudnn_int8_ms`: the same convolution into a virtual tensor, then a
  per-channel scale, then an `int8` output -- the production requantized path.
  One output byte per element.

Each is reported twice: `cudnn_*_over_dispatch` against `best_ms`, an oracle
over every candidate, and `cudnn_*_over_policy` against what the compile-time
policy actually selects.  The policy ratio is the shippable one -- the oracle
had to build and time four candidates to find its answer.

The gap between the two cuDNN columns is what the INT32 contract costs, and it
is strongly shape dependent: with a deep reduction the kernel is bound by the
tensor pipe and the contract is worth a few percent, while with a shallow one
(`1x1` layers) it is bound by the output write and the contract is worth close
to 2x.  Treat `cudnn_int8_over_dispatch` as a measure of that contract, not of
kernel quality.

cuDNN reports no engine configuration for an INT32 output tensor, so exact
parity is unavailable.  Neither cuDNN graph applies a bias; the lab kernels do.
Every engine configuration is checked against the CPU oracle before it is
timed, so a plan that computes the wrong thing can never win.

The legacy `INT8x32` / `NCHW_VECT_C` / `IMPLICIT_PRECOMP_GEMM` path this
replaced is deprecated in cuDNN 9 and has no tensor-core kernel on recent
architectures.  On sm_120 it fell back to a scalar
`cnn::conv2d_grouped_direct_kernel`, ~1.34 ms of GPU time for a 14x14 layer and
the same figure for every shape -- roughly 100x off a real IMMA kernel, which
made any ratio measured against it meaningless.

## Why the stride-2 shapes still trail cuDNN

cuDNN's winning kernel there is a `64x128x64` tile with three staged shared
buffers, and it fills them with `cp.async`: `smsp__inst_executed_op_ldgsts` is
14.9M, all of it the cache-bypass (`.cg`) variant, while
`smsp__inst_executed_op_global_ld` is zero -- it issues no plain global load at
all.  Asynchronous copy stages global to shared with no register live range, so
its third buffer costs 12 KB of shared memory and nothing else.

Hand-written pipelining cannot reproduce that, and both attempts are recorded
in the history.  Two register-staged buffers took the long-scoreboard stall
from 44.9% to 14.4%, close to cuDNN's 11.1%, and still lost, because the extra
slab halved occupancy.  Three buffers lose further: the stride-2 halo is 18.7 KB
per stage, so three of them plus B leave one CTA per SM and every per-tap
barrier drains the whole SM's tensor pipe.  Without asynchronous copies the
latency cover is bounded by the register ring, not by the number of shared
buffers, so the extra buffers buy nothing for their footprint.

CGIR cannot emit `cp.async`, so this gap is a capability question rather than a
tuning one.  Note also that checking one `ldgsts` counter does not establish
whether a kernel uses asynchronous copy: the `.ca` and `.cg` variants have
separate counters and a kernel using one reads zero on the other.

## Measuring shared-memory conflicts

`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st` counts excess
wavefronts, not bank conflicts specifically.  A staging store whose value comes
from a global load that has not yet returned is replayed into the same counter,
so these kernels report an apparent 0-19% conflict rate that tracks warp count
and disappears entirely when the same store addresses are fed a constant.
Padding the layout does not move it, a linear layout does not move it, and it
is not reproducible between runs.

To ask whether a shared-memory layout is actually conflicting, isolate the
access pattern in a kernel that stages from registers rather than from global
memory.  Done that way, both swizzles in this repo report zero.

## Dispatch policy

`dispatch_policy.cuh` turns a `ConvShape` into an algorithm and its tile
parameters:

```cpp
constexpr DispatchDecision choice = select_conv_kernel<Shape>();
launch_by_policy<Shape>(input, filter, bias, output);
```

It is a pure `constexpr` function of the shape, so CGIR can evaluate it at
code-generation time, and `launch_by_policy` instantiates only the chosen
kernel -- a policy build costs one template where the candidate search compiles
eleven.  Every branch re-checks the capability predicate the kernel itself
asserts, so a mis-classified shape falls back to `generic` rather than failing
to compile.  `--strategy policy` runs it; `--strategy both` runs it alongside
the candidates and records its regret.

Each threshold is fitted to a measurement, named in the comment beside it:

- `1x1` convolutions take `generic`.  One tap means no activation to reuse, so
  the halo degenerates to generic plus bookkeeping, and it lost on every
  pointwise shape.
- `spatial_strip` is never selected.  It won 0 of 41 workloads once the halo
  existed, losing even `big_3x3_s1`, the only shape its `out_w % 128` clause
  admits.  It is retained because its predicate documents the 1D reuse.
- Halo `stage_k` is 64 when the staged slab fits 24 KB and 32 otherwise -- the
  footprint that still admits four CTAs per SM, not a rule about stride.
- Split-K fires when the grid cannot fill the machine and the reduction has at
  least two channel groups, with `splits` sized to reach two CTAs per SM.
- The generic tile is the largest shortlisted tile that still fills the
  machine, never wider in N than `gemm_n`, and the smallest tile when the
  reduction is shallow or N is narrow.
- The halo takes `cta_n` 128 at stride greater than one and 64 otherwise.  At
  stride 1 the A slab is small, so there is little to amortise and the halved
  CTA count dominates.

### Retargeting

The policy is a compile-time function, so the machine it targets is described
by macros rather than queried at run time.  Four of them, and every threshold
is derived from them rather than tuned independently:

| macro | default | meaning |
| --- | --- | --- |
| `INT8_LAB_TARGET_SMS` | 36 | SM count; decides whether a grid fills the GPU |
| `INT8_LAB_TARGET_SHARED_PER_SM` | 100 KB | shared-memory carveout per SM |
| `INT8_LAB_TARGET_CTAS_PER_SM` | 4 | occupancy the kernels are tuned for |
| `INT8_LAB_TARGET_WAVES` | 2 | CTA waves before a grid counts as full |

The split-K trigger is `TARGET_WAVES * TARGET_SMS` CTAs and the `stage_k`
footprint budget is `TARGET_SHARED_PER_SM / TARGET_CTAS_PER_SM`, each corrected
by `cta_n / 64` because a `cta_n = 128` tile runs 256-thread CTAs: half as many
are resident, and each may claim twice the shared memory.  Getting that
correction wrong is what made the first `cta_n` integration over-split and
regress `darknet_down_52` by 1.20x.

`scripts/target_flags.py` prints the flags for a device:

```bash
./scripts/target_flags.py --comment                        # this machine
./scripts/target_flags.py --arch sm_75 --sms 68 --ctas-per-sm 3
```

Flags are not sufficient on their own.  The generic tile shortlist and the
`gemm_n <= 32` / `gemm_k <= 128` thresholds were measured on this GPU and need
re-sweeping on another, not just new macros.  The thresholds were also fitted
on `workloads.csv` and measured on the same set, so treat the regret figures as
training-set numbers.  They are mechanistic -- shared-memory footprint, CTA
counts, reduction depth -- rather than curve fits, but re-measure on a new
target.

## Dispatch policy workflow

1. Add representative production shapes to `workloads.csv`.
2. Sweep on each target GPU, CUDA version, and MMA family.
3. Train or hand-author a dispatch policy from the `eligible_*` flags and the
   `*_ms` columns.  Two features carry most of the signal: the CTA count
   (`gemm_m x gemm_n` over the tile size) decides whether split-K pays, and the
   reduction depth (`gemm_k`) decides whether the kernel is bound by the output
   write or by the tensor pipe.
4. Encode the policy outside the kernels; eligibility remains an explicit,
   conservative capability predicate next to each kernel
   (`kSpatialStripEligible`, `kSpatialHaloEligible`).

Eligibility, narrowest first:

- `spatial_strip`: `3x3`, horizontal stride and dilation of one, `C % 64 == 0`,
  `K % 64 == 0`, and `out_w % 128 == 0` so a 128-wide CTA stays inside one
  output row.  That last clause admits exactly one row of `workloads.csv`.
- `spatial_halo`: `C % 64 == 0`, `K % 64 == 0`, and a patch that fits the 48 KB
  static shared-memory allocation.  Arbitrary `R`, `S`, stride, dilation, and
  padding are handled by predication.
- `generic`: `C % 16 == 0`.  It is the fallback and supplies the reference
  comparison for every specialized decision.

A dispatch policy has to be keyed on the MMA family as well as the shape: the
ranking of these three kernels is not the same under `m8n8k16` and
`m16n8k32`.
