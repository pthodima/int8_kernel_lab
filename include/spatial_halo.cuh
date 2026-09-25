#pragma once

#include "conv_shape.cuh"
#include "ptx.cuh"

#include <cuda_runtime.h>

// The halo family generalises the spatial-strip idea from one axis to two.
//
// spatial_strip stages a 1D run of input pixels and reuses it across the S
// filter columns only, which costs `3 * (cta_m + S - 1)` staged pixels per
// R*S taps and forces `out_w % cta_m == 0` so a CTA never leaves an output
// row.  A 2D patch stages `(TileH-1)*stride_h + dilation_h*(R-1) + 1` by
// `(TileW-1)*stride_w + dilation_w*(S-1) + 1` pixels *once* and reuses them
// across all R*S taps, so it moves strictly less activation traffic and drops
// the out_w constraint entirely -- edges are predicated instead.
//
// For 3x3/stride1 with a 4x32 patch that is 6x34 = 204 staged pixels per 128
// outputs per 9 taps, against the strip's 3 * 130 = 390.
template <int TileH, int TileW>
struct HaloTile {
  static constexpr int tile_h = TileH;
  static constexpr int tile_w = TileW;
  static constexpr int cta_m = TileH * TileW;
  static constexpr int cta_n = 64;
  static constexpr int stage_k = 64;
  static constexpr int threads = 128;
  static_assert(cta_m == 128, "fragment and epilogue mapping assume 128 rows");
  static_assert(TileW % 16 == 0,
                "a 16-row MMA fragment must stay inside one patch row");
};

// The patch aspect ratio trades halo overhead against shared-memory footprint
// and is the main thing a dispatch policy should pick per shape: a tall patch
// amortises the vertical halo, a wide one the horizontal halo.
#ifndef INT8_LAB_HALO_TILE_H
#define INT8_LAB_HALO_TILE_H 4
#endif
#ifndef INT8_LAB_HALO_TILE_W
#define INT8_LAB_HALO_TILE_W 32
#endif
using DefaultHaloTile = HaloTile<INT8_LAB_HALO_TILE_H, INT8_LAB_HALO_TILE_W>;

template <class Shape, class Tile = DefaultHaloTile>
struct HaloLayout {
  static constexpr int halo_h =
      (Tile::tile_h - 1) * Shape::stride_h + Shape::dilation_h * (Shape::r - 1) + 1;
  static constexpr int halo_w =
      (Tile::tile_w - 1) * Shape::stride_w + Shape::dilation_w * (Shape::s - 1) + 1;
  static constexpr int halo_pixels = halo_h * halo_w;
  static constexpr int a_bytes = halo_pixels * Tile::stage_k;
  static constexpr int b_bytes = Tile::cta_n * Tile::stage_k;
  static constexpr int shared_bytes = a_bytes + b_bytes;
  static constexpr int a_vectors = a_bytes / 16;
  static constexpr int b_vectors_per_thread = b_bytes / 16 / Tile::threads;
  static constexpr int tiles_x = (Shape::out_w + Tile::tile_w - 1) / Tile::tile_w;
  static constexpr int tiles_y = (Shape::out_h + Tile::tile_h - 1) / Tile::tile_h;
  static constexpr int blocks_n = Shape::gemm_n / Tile::cta_n;
  static constexpr int blocks = Shape::n * tiles_y * tiles_x * blocks_n;
};

// Split-K partitions the channel-group reduction across `Splits` CTAs that
// atomically accumulate into the same output tile.  It exists for shapes where
// `gemm_m x gemm_n` is too small to fill the GPU but `gemm_k` is deep: a 7x7
// output with K=512 launches 16 CTAs onto 36 SMs and then serialises a
// 4608-deep reduction inside each one.  It costs a zeroing pass over the
// output and `Splits` atomics per element, so it only pays when the output is
// small -- which is exactly when the CTA count is too low.
template <class Shape, class Tile = DefaultHaloTile, int Splits = 1>
inline constexpr bool kSplitKUsable =
    Splits >= 1 && Splits <= Shape::c / Tile::stage_k;

// CGIR cannot request dynamic shared memory, so the patch has to fit the 48 KB
// static allocation.  Everything else is handled by predication, which is why
// this predicate is far wider than the strip's.
template <class Shape, class Tile = DefaultHaloTile>
inline constexpr bool kSpatialHaloEligible =
    Shape::c % Tile::stage_k == 0 && Shape::k % Tile::cta_n == 0 &&
    HaloLayout<Shape, Tile>::shared_bytes <= 48 * 1024;

// Same XOR swizzle the strip uses: with 64-byte rows it makes any run of eight
// consecutive rows land on eight disjoint bank quads, so ldmatrix is
// conflict-free regardless of the row the tap slide starts on.
__device__ __forceinline__ int halo_swizzle(int byte_offset) {
  return byte_offset ^ ((byte_offset & 0x1c0) >> 2);
}

template <class Shape, class Tile>
__device__ __forceinline__ void stage_halo_a(int8_t *shared_a,
                                             const int8_t *input, int batch,
                                             int in_y0, int in_x0,
                                             int channel_group, int thread) {
  using L = HaloLayout<Shape, Tile>;
  constexpr int vectors_per_pixel = Tile::stage_k / 16;
#pragma unroll 1
  for (int vector = thread; vector < L::a_vectors; vector += Tile::threads) {
    const int pixel = vector / vectors_per_pixel;
    const int channel =
        channel_group * Tile::stage_k + (vector % vectors_per_pixel) * 16;
    const int hy = pixel / L::halo_w;
    const int y = in_y0 + hy;
    const int x = in_x0 + (pixel - hy * L::halo_w);
    int4 value = make_int4(0, 0, 0, 0);
    if (y >= 0 && y < Shape::h && x >= 0 && x < Shape::w) {
      value = *reinterpret_cast<const int4 *>(
          input + ((batch * Shape::h + y) * Shape::w + x) * Shape::c + channel);
    }
    *reinterpret_cast<int4 *>(shared_a + halo_swizzle(vector * 16)) = value;
  }
}

template <class Shape, class Tile>
struct HaloBPrefetch {
  int4 values[HaloLayout<Shape, Tile>::b_vectors_per_thread];
};

template <class Shape, class Tile>
__device__ __forceinline__ void prefetch_halo_b(
    HaloBPrefetch<Shape, Tile> &prefetch, const int8_t *filter, int n_base,
    int channel_group, int tap, int thread) {
  using L = HaloLayout<Shape, Tile>;
  constexpr int vectors_per_row = Tile::stage_k / 16;
  const int filter_y = tap / Shape::s;
  const int filter_x = tap - filter_y * Shape::s;
#pragma unroll
  for (int i = 0; i < L::b_vectors_per_thread; ++i) {
    const int vector = thread + i * Tile::threads;
    const int output_channel = vector / vectors_per_row;
    const int channel =
        channel_group * Tile::stage_k + (vector % vectors_per_row) * 16;
    prefetch.values[i] = *reinterpret_cast<const int4 *>(
        filter + (((n_base + output_channel) * Shape::r + filter_y) * Shape::s +
                  filter_x) * Shape::c + channel);
  }
}

template <class Shape, class Tile>
__device__ __forceinline__ void store_halo_b(
    int8_t *shared_b, const HaloBPrefetch<Shape, Tile> &prefetch, int thread) {
  using L = HaloLayout<Shape, Tile>;
#pragma unroll
  for (int i = 0; i < L::b_vectors_per_thread; ++i) {
    const int vector = thread + i * Tile::threads;
    *reinterpret_cast<int4 *>(shared_b + halo_swizzle(vector * 16)) =
        prefetch.values[i];
  }
}

// Byte offset of the staged input pixel feeding output row `m_local` for tap
// (filter_y, filter_x).  This is the halo's equivalent of the strip's
// `a_base + filter_x` slide; the only difference is that the patch row moves
// too, so the tap walks both axes.
template <class Shape, class Tile>
__device__ __forceinline__ int halo_a_offset(int m_local, int filter_y,
                                             int filter_x, int k_step) {
  using L = HaloLayout<Shape, Tile>;
  const int oy_local = m_local / Tile::tile_w;
  const int ox_local = m_local - oy_local * Tile::tile_w;
  const int hy = oy_local * Shape::stride_h + filter_y * Shape::dilation_h;
  const int hx = ox_local * Shape::stride_w + filter_x * Shape::dilation_w;
  return (hy * L::halo_w + hx) * Tile::stage_k + k_step;
}

#if INT8_LAB_MMA_SM80
template <class Shape, class Tile = DefaultHaloTile, int Splits = 1>
__global__ __launch_bounds__(Tile::threads, INT8_LAB_MIN_CTAS_PER_SM) void spatial_halo_sm80_kernel(
    const int8_t *__restrict__ input, const int8_t *__restrict__ filter,
    const int32_t *__restrict__ bias, int32_t *__restrict__ output) {
  using L = HaloLayout<Shape, Tile>;
  __shared__ __align__(16) int8_t shared[L::shared_bytes];
  int8_t *shared_a = shared;
  int8_t *shared_b = shared + L::a_bytes;
  const int thread = threadIdx.x, lane = thread & 31, warp = thread >> 5;

  int block = blockIdx.x;
  const int n_base = (block % L::blocks_n) * Tile::cta_n;
  block /= L::blocks_n;
  const int tile_ox = (block % L::tiles_x) * Tile::tile_w;
  block /= L::tiles_x;
  const int tile_oy = (block % L::tiles_y) * Tile::tile_h;
  block /= L::tiles_y;
  const int batch = block % Shape::n;
  const int split = block / Shape::n;

  constexpr int channel_groups = Shape::c / Tile::stage_k;
  constexpr int groups_per_split = (channel_groups + Splits - 1) / Splits;
  const int cg_begin = split * groups_per_split;
  const int cg_end = cg_begin + groups_per_split < channel_groups
                         ? cg_begin + groups_per_split
                         : channel_groups;

  const int in_y0 = tile_oy * Shape::stride_h - Shape::pad_h;
  const int in_x0 = tile_ox * Shape::stride_w - Shape::pad_w;
  const int warp_m = (warp >> 1) * 64, warp_n = (warp & 1) * 32;
  constexpr int taps = Shape::r * Shape::s;
  int32_t acc[4][4][4] = {};

#pragma unroll 1
  for (int cg = cg_begin; cg < cg_end; ++cg) {
    stage_halo_a<Shape, Tile>(shared_a, input, batch, in_y0, in_x0, cg, thread);
    __syncthreads();
    HaloBPrefetch<Shape, Tile> prefetched;
    prefetch_halo_b<Shape, Tile>(prefetched, filter, n_base, cg, 0, thread);
    store_halo_b<Shape, Tile>(shared_b, prefetched, thread);
    __syncthreads();

#pragma unroll
    for (int tap = 0; tap < taps; ++tap) {
      const int filter_y = tap / Shape::s, filter_x = tap % Shape::s;
      if (tap + 1 < taps) {
        prefetch_halo_b<Shape, Tile>(prefetched, filter, n_base, cg, tap + 1,
                                     thread);
      }
#pragma unroll
      for (int k = 0; k < Tile::stage_k; k += 32) {
        uint32_t a[4][4], b[4][2];
#pragma unroll
        for (int m = 0; m < 4; ++m) {
          load_matrix_x4(
              shared_a + halo_swizzle(halo_a_offset<Shape, Tile>(
                             warp_m + m * 16 + lane % 16, filter_y, filter_x,
                             k + (lane / 16) * 16)),
              a[m][0], a[m][1], a[m][2], a[m][3]);
        }
#pragma unroll
        for (int n = 0; n < 4; ++n) {
          load_matrix_x2(shared_b + halo_swizzle((warp_n + n * 8 + lane % 8) *
                                                     Tile::stage_k +
                                                 k + ((lane / 8) % 2) * 16),
                         b[n][0], b[n][1]);
        }
#pragma unroll
        for (int m = 0; m < 4; ++m)
#pragma unroll
          for (int n = 0; n < 4; ++n)
            mma_m16n8k32(acc[m][n][0], acc[m][n][1], acc[m][n][2], acc[m][n][3],
                         a[m][0], a[m][1], a[m][2], a[m][3], b[n][0], b[n][1]);
      }
      __syncthreads();
      if (tap + 1 < taps) {
        store_halo_b<Shape, Tile>(shared_b, prefetched, thread);
        __syncthreads();
      }
    }
  }

#pragma unroll
  for (int m = 0; m < 4; ++m)
#pragma unroll
    for (int n = 0; n < 4; ++n) {
      const int col = n_base + warp_n + n * 8 + (lane & 3) * 2;
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int m_local = warp_m + m * 16 + lane / 4 + half * 8;
        const int oy = tile_oy + m_local / Tile::tile_w;
        const int ox = tile_ox + m_local % Tile::tile_w;
        if (oy < Shape::out_h && ox < Shape::out_w) {
          const int row = (batch * Shape::out_h + oy) * Shape::out_w + ox;
          if constexpr (Splits == 1) {
            store_bias_pair<Shape::gemm_n>(output, bias, row, col,
                                           acc[m][n][half * 2],
                                           acc[m][n][half * 2 + 1]);
          } else {
            // The launcher zeroes the output, so exactly one split contributes
            // the bias and every split contributes its partial sum.
            int32_t *dst = output + static_cast<size_t>(row) * Shape::gemm_n + col;
            const int32_t b0 = split == 0 ? bias[col] : 0;
            const int32_t b1 = split == 0 ? bias[col + 1] : 0;
            atomicAdd(dst, acc[m][n][half * 2] + b0);
            atomicAdd(dst + 1, acc[m][n][half * 2 + 1] + b1);
          }
        }
      }
    }
}
#else
template <class Shape, class Tile = DefaultHaloTile, int Splits = 1>
__global__ __launch_bounds__(Tile::threads, INT8_LAB_MIN_CTAS_PER_SM) void spatial_halo_sm75_kernel(
    const int8_t *__restrict__ input, const int8_t *__restrict__ filter,
    const int32_t *__restrict__ bias, int32_t *__restrict__ output) {
  using L = HaloLayout<Shape, Tile>;
  __shared__ __align__(16) int8_t shared[L::shared_bytes];
  int8_t *shared_a = shared;
  int8_t *shared_b = shared + L::a_bytes;
  const int thread = threadIdx.x, lane = thread & 31, warp = thread >> 5;

  int block = blockIdx.x;
  const int n_base = (block % L::blocks_n) * Tile::cta_n;
  block /= L::blocks_n;
  const int tile_ox = (block % L::tiles_x) * Tile::tile_w;
  block /= L::tiles_x;
  const int tile_oy = (block % L::tiles_y) * Tile::tile_h;
  block /= L::tiles_y;
  const int batch = block % Shape::n;
  const int split = block / Shape::n;

  constexpr int channel_groups = Shape::c / Tile::stage_k;
  constexpr int groups_per_split = (channel_groups + Splits - 1) / Splits;
  const int cg_begin = split * groups_per_split;
  const int cg_end = cg_begin + groups_per_split < channel_groups
                         ? cg_begin + groups_per_split
                         : channel_groups;

  const int in_y0 = tile_oy * Shape::stride_h - Shape::pad_h;
  const int in_x0 = tile_ox * Shape::stride_w - Shape::pad_w;
  const int warp_m_group = (warp >> 1) << 3, warp_n_group = (warp & 1) << 3;
  constexpr int taps = Shape::r * Shape::s;
  int32_t acc[8][4][2] = {};

#pragma unroll 1
  for (int cg = cg_begin; cg < cg_end; ++cg) {
    stage_halo_a<Shape, Tile>(shared_a, input, batch, in_y0, in_x0, cg, thread);
    __syncthreads();
    HaloBPrefetch<Shape, Tile> prefetched;
    prefetch_halo_b<Shape, Tile>(prefetched, filter, n_base, cg, 0, thread);
    store_halo_b<Shape, Tile>(shared_b, prefetched, thread);
    __syncthreads();

#pragma unroll
    for (int tap = 0; tap < taps; ++tap) {
      const int filter_y = tap / Shape::s, filter_x = tap % Shape::s;
      if (tap + 1 < taps) {
        prefetch_halo_b<Shape, Tile>(prefetched, filter, n_base, cg, tap + 1,
                                     thread);
      }
#pragma unroll
      for (int k_step = 0; k_step < Tile::stage_k; k_step += 16) {
        uint32_t a[8], b[4];
#pragma unroll
        for (int group = 0; group < 2; ++group) {
          const int source_m8 = group * 4 + (lane >> 3);
          load_matrix_x4(
              shared_a + halo_swizzle(halo_a_offset<Shape, Tile>(
                             warp_m_group + (lane & 7) + source_m8 * 16,
                             filter_y, filter_x, k_step)),
              a[group * 4], a[group * 4 + 1], a[group * 4 + 2], a[group * 4 + 3]);
        }
        load_matrix_x4(shared_b + halo_swizzle((warp_n_group + (lane & 7)) *
                                                   Tile::stage_k +
                                               (lane >> 3) * 16 * Tile::stage_k +
                                               k_step),
                       b[0], b[1], b[2], b[3]);
#pragma unroll
        for (int mma_m = 0; mma_m < 8; ++mma_m)
#pragma unroll
          for (int mma_n = 0; mma_n < 4; ++mma_n)
            mma_m8n8k16(acc[mma_m][mma_n][0], acc[mma_m][mma_n][1], a[mma_m],
                        b[mma_n]);
      }
      __syncthreads();
      if (tap + 1 < taps) {
        store_halo_b<Shape, Tile>(shared_b, prefetched, thread);
        __syncthreads();
      }
    }
  }

#pragma unroll
  for (int mma_m = 0; mma_m < 8; ++mma_m)
#pragma unroll
    for (int mma_n = 0; mma_n < 4; ++mma_n) {
      const int m_local = warp_m_group + lane / 4 + mma_m * 16;
      const int oy = tile_oy + m_local / Tile::tile_w;
      const int ox = tile_ox + m_local % Tile::tile_w;
      const int col = n_base + warp_n_group + (lane & 3) * 2 + mma_n * 16;
      if (oy < Shape::out_h && ox < Shape::out_w) {
        const int row = (batch * Shape::out_h + oy) * Shape::out_w + ox;
        if constexpr (Splits == 1) {
          store_bias_pair<Shape::gemm_n>(output, bias, row, col,
                                         acc[mma_m][mma_n][0],
                                         acc[mma_m][mma_n][1]);
        } else {
          int32_t *dst = output + static_cast<size_t>(row) * Shape::gemm_n + col;
          atomicAdd(dst, acc[mma_m][mma_n][0] + (split == 0 ? bias[col] : 0));
          atomicAdd(dst + 1, acc[mma_m][mma_n][1] + (split == 0 ? bias[col + 1] : 0));
        }
      }
    }
}
#endif

template <class Shape, class Tile = DefaultHaloTile, int Splits = 1>
void launch_spatial_halo(const int8_t *input, const int8_t *filter,
                         const int32_t *bias, int32_t *output,
                         cudaStream_t stream = 0) {
  static_assert(kSpatialHaloEligible<Shape, Tile>,
                "shape is not eligible for the spatial halo tile");
  static_assert(kSplitKUsable<Shape, Tile, Splits>,
                "Splits must not exceed the channel-group count");
  constexpr int blocks = HaloLayout<Shape, Tile>::blocks * Splits;
  if constexpr (Splits > 1) {
    cudaMemsetAsync(output, 0, Shape::output_elements * sizeof(int32_t), stream);
  }
#if INT8_LAB_MMA_SM80
  spatial_halo_sm80_kernel<Shape, Tile, Splits>
      <<<blocks, Tile::threads, 0, stream>>>(input, filter, bias, output);
#else
  spatial_halo_sm75_kernel<Shape, Tile, Splits>
      <<<blocks, Tile::threads, 0, stream>>>(input, filter, bias, output);
#endif
}
