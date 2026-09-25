#pragma once

// A compile-time dispatch policy: given a ConvShape, choose the algorithm and
// its tile parameters.
//
// Every rule below is a threshold fitted to a measurement in this repo, not a
// guess, and the comment on each says which.  The policy is a pure function of
// the shape so CGIR can evaluate it at code-generation time, and only the
// chosen kernel is instantiated, so a build costs one template rather than the
// ten the candidate search compiles.
//
// It is deliberately conservative: every branch re-checks the same capability
// predicate the kernel itself asserts, so a shape the policy mis-classifies
// falls back to `generic` rather than failing to compile.

#include "conv_shape.cuh"
#include "generic_implicit_gemm.cuh"
#include "spatial_halo.cuh"
#include "spatial_strip.cuh"

#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Machine description.  Every threshold below is derived from these four
// numbers rather than being a tuned constant, so retargeting is a matter of
// passing the right -D flags.  `scripts/target_flags.py` queries a device and
// prints them.  The defaults describe the RTX 5060 Ti (sm_120) this repo was
// measured on; the commented values are an RTX 2080 Ti (sm_75) for contrast.
//
// Only INT8_LAB_TARGET_SMS affects whether the grid fills the machine; the
// other three set how much of an SM one CTA may claim.
#ifndef INT8_LAB_TARGET_SMS
#define INT8_LAB_TARGET_SMS 36            // 2080 Ti: 68
#endif
#ifndef INT8_LAB_TARGET_SHARED_PER_SM
#define INT8_LAB_TARGET_SHARED_PER_SM (100 * 1024)   // 2080 Ti: 64 * 1024
#endif
// How many CTAs of the baseline 128-thread shape should fit on one SM.  It is
// the occupancy the kernels are tuned for, and it sets both the shared-memory
// budget per CTA and the ptxas register hint.
#ifndef INT8_LAB_TARGET_CTAS_PER_SM
#define INT8_LAB_TARGET_CTAS_PER_SM 4     // m8n8k16 spills at 4; use 3
#endif
// Waves of CTAs wanted before the grid counts as filling the machine.  Below
// this, split-K is worth its zeroing pass and atomics.
#ifndef INT8_LAB_TARGET_WAVES
#define INT8_LAB_TARGET_WAVES 2
#endif

enum class ConvAlgorithm {
  kGeneric,
  kSpatialStrip,
  kSpatialHalo,
  kSpatialHaloSplitK,
};

struct DispatchDecision {
  ConvAlgorithm algorithm = ConvAlgorithm::kGeneric;
  int halo_tile_h = 4;
  int halo_tile_w = 32;
  int halo_stage_k = 64;
  int halo_splits = 1;
  int halo_cta_n = 64;
  int cta_m = 64;
  int cta_n = 64;
  int warp_m = 32;
  int warp_n = 16;
  int stage_k = 64;
};

constexpr int policy_ceil_div(int a, int b) { return (a + b - 1) / b; }

constexpr int policy_round_up_pow2(int v) {
  int r = 1;
  while (r < v) r *= 2;
  return r;
}

template <class Shape>
constexpr DispatchDecision select_conv_kernel() {
  DispatchDecision d{};
  constexpr int kTargetCtas = INT8_LAB_TARGET_WAVES * INT8_LAB_TARGET_SMS;
  // The driver instantiates split counts up to this; keep them in step.
  constexpr int kMaxSplits = 16;

  // ---- generic tile, which is also the fallback -------------------------
  // Sweeping 294 tiles over five shapes: every tile in the top eight used
  // stage_k 64 and a warp tile of 16 or 32 rows, while the old 64x16 warp tile
  // ranked 8.1% against 0.9% for 32x16.  A 64-row warp tile forces eight A
  // fragments per warp and only four warps over the CTA.
  //
  // 128x64 is deliberately absent: it ranked 4.3% and lost on all six shapes
  // it was re-measured against, by up to 1.41x.  Nothing selects it.
  struct GenericChoice { int cm, cn, wm, wn, sk; };
  constexpr GenericChoice kShortlist[] = {
      {64, 64, 32, 16, 64},
      {64, 32, 16, 16, 64},
      {32, 32, 16, 16, 64},
  };
  constexpr int kShortlistSize = sizeof(kShortlist) / sizeof(kShortlist[0]);

  // Never tile N wider than the problem: darknet_stem_padded has gemm_n 32, so
  // a 64-wide tile throws away half of every MMA.  Entries are ordered large
  // to small, so the first survivor is the largest that fits.
  int first = 0;
  while (first < kShortlistSize - 1 && kShortlist[first].cn > Shape::gemm_n)
    ++first;

  // A shallow reduction cannot amortise a large staged tile -- the prologue
  // and epilogue dominate -- and a narrow one cannot fill a wide warp tile.
  // Measured: resnet_s2_expand (gemm_k 64) 7.3 -> 5.2 us and
  // darknet_stem_padded (gemm_n 32) 87.4 -> 58.5 us on the smallest tile,
  // while batch16_bottleneck_reduce (gemm_k 256, gemm_n 64) wants the largest.
  const bool prefer_small = Shape::gemm_n <= 32 || Shape::gemm_k <= 128;

  int chosen = kShortlistSize - 1;
  if (!prefer_small) {
    for (int i = first; i < kShortlistSize; ++i) {
      const int ctas = policy_ceil_div(Shape::gemm_m, kShortlist[i].cm) *
                       policy_ceil_div(Shape::gemm_n, kShortlist[i].cn);
      if (ctas >= kTargetCtas) { chosen = i; break; }
    }
  }
  d.cta_m = kShortlist[chosen].cm;
  d.cta_n = kShortlist[chosen].cn;
  d.warp_m = kShortlist[chosen].wm;
  d.warp_n = kShortlist[chosen].wn;
  d.stage_k = kShortlist[chosen].sk;

  // ---- is a spatial kernel worth it at all? -----------------------------
  // A 1x1 convolution has one tap, so there is no activation to reuse across
  // taps and the halo degenerates to generic plus halo bookkeeping.  It lost
  // on every pointwise shape in the 41-case sweep -- resnet_s2_expand,
  // mobilenet_pointwise, darknet_res_13_reduce, batch16_bottleneck_reduce.
  if (Shape::r == 1 && Shape::s == 1) return d;

  // spatial_strip is never selected.  It is retained as a candidate because
  // its eligibility predicate documents the 1D form of the reuse, but it won
  // 0 of 41 workloads once the halo existed, losing even big_3x3_s1, the only
  // shape its out_w % 128 clause admits.
  // ---- cta_n: wider N amortises the A slab, at the cost of half the CTAs ----
  // The A slab does not depend on cta_n, so doubling it buys 2x the MMA per
  // A-stage for about 10% more shared memory -- but it also halves the grid.
  // Measured over workloads.csv, that trade splits exactly on stride: every
  // stride>1 shape gains (big_3x3_s2 1.22x, resnet_s4_down 1.14x,
  // resnet_s5_down 1.11x, big_5x5_s2 1.06x) and every stride-1 3x3 loses
  // (darknet_res_52_expand 0.81x, dilated_3x3 0.82x, batch8_3x3 0.84x).  At
  // stride 1 the slab is small, so there is little to amortise and the lost
  // CTAs dominate.
  constexpr bool kWide =
      (Shape::stride_h > 1 || Shape::stride_w > 1) && Shape::k % 128 == 0;
  constexpr int kCtaN = kWide ? 128 : 64;
  d.halo_cta_n = kCtaN;
  using Halo64 = HaloTile<4, 32, 64, kCtaN>;
  using Halo32 = HaloTile<4, 32, 32, kCtaN>;
  constexpr bool halo64 = kSpatialHaloEligible<Shape, Halo64>;
  constexpr bool halo32 = kSpatialHaloEligible<Shape, Halo32>;
  if (!halo64 && !halo32) return d;

  // ---- stage_k: footprint, not stride -----------------------------------
  // Stride and filter size inflate the halo until the staged slab rather than
  // the register file binds occupancy.  A 4x32 patch at stride 2 needs 41 KB
  // and gets 2 CTAs/SM at stage_k 64, against 21 KB and 4 at 32; halving it
  // was worth 1.55x on big_3x3_s2 and made big_5x5_s2 eligible at all, while
  // costing 10% on big_3x3_s1 where the patch is already compact.  The
  // threshold is the footprint that still admits four CTAs per SM.
  // The threshold is a warps-per-SM target, not a CTA count: a cta_n=128 tile
  // runs 256-thread CTAs, so two of them occupy the same sixteen warps that
  // four 128-thread CTAs do, and it may spend twice the shared memory to get
  // there.  Scaling by cta_n/64 is what lets big_3x3_s2 keep stage_k 64 at
  // cta_n 128 (4.92 ms) instead of dropping to 32 (5.33 ms).
  // Shared memory one CTA may claim and still hit the occupancy target.  The
  // cta_n/64 factor is the residency correction: a cta_n=128 tile runs
  // 256-thread CTAs, so half as many fit and each may spend twice as much.
  constexpr int kFootprintBudget =
      (INT8_LAB_TARGET_SHARED_PER_SM / INT8_LAB_TARGET_CTAS_PER_SM) *
      (kCtaN / 64);
  const bool compact =
      halo64 && HaloLayout<Shape, Halo64>::shared_bytes <= kFootprintBudget;
  d.halo_stage_k = (compact || !halo32) ? 64 : 32;
  const int stage_k = d.halo_stage_k;

  // ---- split-K: only when the grid cannot fill the machine ---------------
  // Split-K trades a zeroing pass and Splits atomics per output element for
  // CTA count, so it pays exactly when gemm_m x gemm_n is too small to fill
  // the GPU and gemm_k is deep enough to divide.  resnet_s5_3x3 launched 16
  // CTAs onto 36 SMs and then serialised a 4608-deep reduction in each; it
  // wants eight ways.  Large outputs make the zeroing pass dominate, which is
  // why big_1x1_s1 and big_3x3_s1 regress 4x under it.
  const int tiles_x = policy_ceil_div(Shape::out_w, d.halo_tile_w);
  const int tiles_y = policy_ceil_div(Shape::out_h, d.halo_tile_h);
  const int ctas = Shape::n * tiles_y * tiles_x * (Shape::k / kCtaN);
  const int channel_groups = Shape::c / stage_k;
  // The target scales with cta_n for the same reason the footprint budget
  // does.  kTargetCtas was calibrated against 128-thread CTAs, four resident
  // per SM; a cta_n=128 tile runs 256-thread CTAs, only two fit, and each does
  // twice the work, so the machine is filled by half as many.  Comparing the
  // smaller grid against the unscaled target over-splits: darknet_down_52 took
  // eight ways where two suffice (29.7 -> 33.8 us) and resnet_s5_down sixteen
  // where the oracle wants eight.
  const int target = kTargetCtas * 64 / kCtaN;
  if (ctas < target && channel_groups >= 2) {
    int splits = policy_round_up_pow2(policy_ceil_div(target, ctas));
    if (splits > channel_groups) splits = channel_groups;
    if (splits > kMaxSplits) splits = kMaxSplits;
    if (splits >= 2) {
      d.algorithm = ConvAlgorithm::kSpatialHaloSplitK;
      d.halo_splits = splits;
      return d;
    }
  }
  d.algorithm = ConvAlgorithm::kSpatialHalo;
  return d;
}

// Launches whatever the policy chose.  `if constexpr` keeps the branches that
// were not chosen uninstantiated, so an ineligible configuration never reaches
// a kernel's static_assert.
template <class Shape>
void launch_by_policy(const int8_t *input, const int8_t *filter,
                      const int32_t *bias, int32_t *output,
                      cudaStream_t stream = 0) {
  constexpr DispatchDecision kChoice = select_conv_kernel<Shape>();
  using HTile = HaloTile<kChoice.halo_tile_h, kChoice.halo_tile_w,
                         kChoice.halo_stage_k, kChoice.halo_cta_n>;
  using GTile = GenericTile<kChoice.cta_m, kChoice.cta_n, kChoice.warp_m,
                            kChoice.warp_n, kChoice.stage_k>;
  if constexpr (kChoice.algorithm == ConvAlgorithm::kSpatialHaloSplitK) {
    launch_spatial_halo<Shape, HTile, kChoice.halo_splits>(input, filter, bias,
                                                           output, stream);
  } else if constexpr (kChoice.algorithm == ConvAlgorithm::kSpatialHalo) {
    launch_spatial_halo<Shape, HTile, 1>(input, filter, bias, output, stream);
  } else {
#if INT8_LAB_MMA_SM80
    launch_generic_sm80<Shape, GTile>(input, filter, bias, output, stream);
#else
    launch_generic<Shape, GTile>(input, filter, bias, output, stream);
#endif
  }
}

inline const char *algorithm_name(ConvAlgorithm algorithm) {
  switch (algorithm) {
    case ConvAlgorithm::kGeneric: return "generic";
    case ConvAlgorithm::kSpatialStrip: return "strip";
    case ConvAlgorithm::kSpatialHalo: return "halo";
    case ConvAlgorithm::kSpatialHaloSplitK: return "halo_splitk";
  }
  return "unknown";
}
