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

Two kernel families have the same input/output contract:

- `generic`: staged implicit GEMM.  It covers all listed shapes with channel
  counts divisible by 16.
- `spatial_strip`: a `128 x 64 x 64` SM75 specialization.  It reuses a
  horizontally contiguous input strip for 3x3, horizontal-stride-one
  convolutions.  Unsupported shapes are reported as ineligible, rather than
  silently using a different algorithm.

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

Use `--strategy generic` or `--strategy strip` to isolate a candidate.

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

On the RTX 2080 Ti host, build and write a combined dispatch/cuDNN CSV with:

```bash
./scripts/sweep.py --arch sm_75 --with-cudnn \
    --output build/dispatch_vs_cudnn.csv
```

This uses cuDNN's legacy `INT8x32` `NCHW_VECT_C` +
`IMPLICIT_PRECOMP_GEMM` path, with filter reorder and descriptor setup outside
the timed region.  Eligible rows record `dispatch_ms`, `cudnn_ms`, and
`cudnn_over_dispatch`; `C % 32 != 0` or `K % 32 != 0` rows are retained with
an explicit ineligible contract.  The comparison is performance-only:
the dispatch kernel writes INT32 while this SM75 cuDNN baseline writes INT8.

## Dispatch policy workflow

1. Add representative production shapes to `workloads.csv`.
2. Sweep on each target GPU and CUDA version.
3. Train or hand-author a dispatch policy from `eligible_strip`,
   `generic_ms`, and `strip_ms`.
4. Encode the policy outside the kernels; eligibility remains an explicit,
   conservative capability predicate in `spatial_strip.cuh`.

The strip eligibility is intentionally narrow: `3x3`, horizontal stride and
dilation of one, `C % 64 == 0`, `K % 64 == 0`, and a 128-wide CTA that remains
inside one output row.  The generic candidate is the fallback and supplies the
reference comparison for each strip decision.
