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

// The machine width the policy targets.  It only enters through "are there
// enough CTAs to fill the GPU", so it wants the SM count of the deployment
// target, not of the build host.
#ifndef INT8_LAB_TARGET_SMS
#define INT8_LAB_TARGET_SMS 36
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
  constexpr int kTargetCtas = 2 * INT8_LAB_TARGET_SMS;

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
  using Halo64 = HaloTile<4, 32, 64>;
  using Halo32 = HaloTile<4, 32, 32>;
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
  constexpr int kFootprintForFourCtas = 24 * 1024;
  const bool compact =
      halo64 && HaloLayout<Shape, Halo64>::shared_bytes <= kFootprintForFourCtas;
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
  const int ctas = Shape::n * tiles_y * tiles_x * (Shape::k / 64);
  const int channel_groups = Shape::c / stage_k;
  if (ctas < kTargetCtas && channel_groups >= 2) {
    int splits = policy_round_up_pow2(policy_ceil_div(kTargetCtas, ctas));
    if (splits > channel_groups) splits = channel_groups;
    if (splits > 16) splits = 16;
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
                         kChoice.halo_stage_k>;
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
