#pragma once

#include "conv_shape.cuh"
#include "ptx.cuh"

#include <cuda_runtime.h>

// A compact, intentionally conservative implicit-GEMM baseline.  Four warps
// compute a 64x64 CTA tile; shared A is reused by every warp and each warp
// owns a 16-column B tile.  It has the same INT32 contract as spatial strip.
template <int StageK = 32>
struct GenericTile {
  static constexpr int cta_m = 64;
  static constexpr int cta_n = 64;
  static constexpr int warp_m = 64;
  static constexpr int warp_n = 16;
  static constexpr int warps = 4;
  static constexpr int threads = warps * 32;
  static constexpr int stage_k = StageK;
  static constexpr int mma_m = 8;
  static constexpr int mma_n = 8;
  static constexpr int mma_k = 16;
  static_assert(StageK % mma_k == 0);
};

template <int StageK>
__device__ __forceinline__ int generic_smem_offset(int row, int inner) {
  constexpr int vectors_per_row = StageK / 16;
  int vector = inner / 16;
  int group = row / 8;
  int row_in_group = row & 7;
  int rotation = vector * (8 / vectors_per_row);
  return (group * 8 * vectors_per_row + vector * 8 +
          (row_in_group ^ rotation)) * 16;
}

template <class Shape, class Tile>
__device__ __forceinline__ void generic_stage(
    const int8_t *input, const int8_t *filter, int8_t *shared, int tile_m,
    int tile_n, int k_base, int thread, int warp) {
  constexpr int a_vectors = Tile::cta_m * Tile::stage_k / 16;
  constexpr int b_vectors_per_warp = Tile::warp_n * Tile::stage_k / 16;
  int8_t *shared_a = shared;
  int8_t *shared_b = shared_a + Tile::cta_m * Tile::stage_k;

  for (int vector = thread; vector < a_vectors; vector += Tile::threads) {
    const int row = vector / (Tile::stage_k / 16);
    const int inner = k_base + (vector % (Tile::stage_k / 16)) * 16;
    const int m = tile_m + row;
    uint4 value = make_uint4(0, 0, 0, 0);
    if (m < Shape::gemm_m && inner < Shape::gemm_k) {
      const int ox = m % Shape::out_w;
      const int oy = (m / Shape::out_w) % Shape::out_h;
      const int batch = m / (Shape::out_h * Shape::out_w);
      const int channel = inner % Shape::c;
      const int fx = (inner / Shape::c) % Shape::s;
      const int fy = (inner / Shape::c / Shape::s) % Shape::r;
      const int ix = ox * Shape::stride_w + fx * Shape::dilation_w - Shape::pad_w;
      const int iy = oy * Shape::stride_h + fy * Shape::dilation_h - Shape::pad_h;
      if (ix >= 0 && ix < Shape::w && iy >= 0 && iy < Shape::h) {
        value = *reinterpret_cast<const uint4 *>(
            input + ((batch * Shape::h + iy) * Shape::w + ix) * Shape::c +
            channel);
      }
    }
    *reinterpret_cast<uint4 *>(
        shared_a + generic_smem_offset<Tile::stage_k>(row, inner - k_base)) =
        value;
  }

  int8_t *warp_b = shared_b + warp * Tile::warp_n * Tile::stage_k;
  for (int vector = thread & 31; vector < b_vectors_per_warp; vector += 32) {
    const int row = vector / (Tile::stage_k / 16);
    const int inner = k_base + (vector % (Tile::stage_k / 16)) * 16;
    const int n = tile_n + warp * Tile::warp_n + row;
    uint4 value = make_uint4(0, 0, 0, 0);
    if (n < Shape::gemm_n && inner < Shape::gemm_k) {
      const int channel = inner % Shape::c;
      const int fx = (inner / Shape::c) % Shape::s;
      const int fy = (inner / Shape::c / Shape::s) % Shape::r;
      value = *reinterpret_cast<const uint4 *>(
          filter + ((n * Shape::r + fy) * Shape::s + fx) * Shape::c + channel);
    }
    *reinterpret_cast<uint4 *>(
        warp_b + generic_smem_offset<Tile::stage_k>(row, inner - k_base)) =
        value;
  }
}

template <class Shape, class Tile = GenericTile<>>
__global__ __launch_bounds__(Tile::threads, 1) void generic_implicit_gemm_kernel(
    const int8_t *__restrict__ input, const int8_t *__restrict__ filter,
    const int32_t *__restrict__ bias, int32_t *__restrict__ output) {
  __shared__ __align__(16) int8_t shared[
      (Tile::cta_m + Tile::cta_n) * Tile::stage_k];
  const int thread = threadIdx.x;
  const int warp = thread >> 5;
  const int lane = thread & 31;
  const int blocks_n = (Shape::gemm_n + Tile::cta_n - 1) / Tile::cta_n;
  const int tile_m = (blockIdx.x / blocks_n) * Tile::cta_m;
  const int tile_n = (blockIdx.x % blocks_n) * Tile::cta_n;
  const int warp_n = tile_n + warp * Tile::warp_n;
  int32_t accum[8][2][2] = {};

  for (int k_base = 0; k_base < Shape::gemm_k; k_base += Tile::stage_k) {
    generic_stage<Shape, Tile>(input, filter, shared, tile_m, tile_n, k_base,
                               thread, warp);
    __syncthreads();
    if (tile_m < Shape::gemm_m && warp_n < Shape::gemm_n) {
      int8_t *shared_a = shared;
      int8_t *shared_b =
          shared_a + Tile::cta_m * Tile::stage_k +
          warp * Tile::warp_n * Tile::stage_k;
#pragma unroll
      for (int k = 0; k < Tile::stage_k; k += Tile::mma_k) {
        uint32_t a[8];
        uint32_t b[2];
#pragma unroll
        for (int m = 0; m < 8; m += 2) {
          load_matrix_x2(shared_a + generic_smem_offset<Tile::stage_k>(
                                       m * Tile::mma_m + lane % 16, k),
                         a[m], a[m + 1]);
        }
        load_matrix_x2(
            shared_b + generic_smem_offset<Tile::stage_k>(lane % 16, k),
            b[0], b[1]);
#pragma unroll
        for (int m = 0; m < 8; ++m) {
#pragma unroll
          for (int n = 0; n < 2; ++n) {
            mma_m8n8k16(accum[m][n][0], accum[m][n][1], a[m], b[n]);
          }
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int m = 0; m < 8; ++m) {
#pragma unroll
    for (int n = 0; n < 2; ++n) {
      const int row = tile_m + m * 8 + lane / 4;
      const int col = warp_n + n * 8 + (lane & 3) * 2;
      if (row < Shape::gemm_m) {
        if (col + 1 < Shape::gemm_n) {
          store_bias_pair<Shape::gemm_n>(output, bias, row, col, accum[m][n][0],
                                         accum[m][n][1]);
        } else if (col < Shape::gemm_n) {
          output[static_cast<size_t>(row) * Shape::gemm_n + col] =
              accum[m][n][0] + bias[col];
        }
      }
    }
  }
}

template <class Shape>
void launch_generic(const int8_t *input, const int8_t *filter,
                    const int32_t *bias, int32_t *output,
                    cudaStream_t stream = 0) {
  using Tile = GenericTile<>;
  constexpr int blocks_m = (Shape::gemm_m + Tile::cta_m - 1) / Tile::cta_m;
  constexpr int blocks_n = (Shape::gemm_n + Tile::cta_n - 1) / Tile::cta_n;
  generic_implicit_gemm_kernel<Shape, Tile>
      <<<blocks_m * blocks_n, Tile::threads, 0, stream>>>(input, filter, bias,
                                                            output);
}

#if INT8_LAB_MMA_SM80
// This is the CGIR SM80 fragment contract: A has four U32 registers, B has
// two, C has four; lane subscript K offsets are 0 or 16.
template <class Shape, class Tile = GenericTile<>>
__global__ __launch_bounds__(Tile::threads, 1) void generic_implicit_gemm_sm80_kernel(
    const int8_t *__restrict__ input, const int8_t *__restrict__ filter,
    const int32_t *__restrict__ bias, int32_t *__restrict__ output) {
  __shared__ __align__(16) int8_t shared[
      (Tile::cta_m + Tile::cta_n) * Tile::stage_k];
  const int thread = threadIdx.x, warp = thread >> 5, lane = thread & 31;
  const int blocks_n = (Shape::gemm_n + Tile::cta_n - 1) / Tile::cta_n;
  const int tile_m = (blockIdx.x / blocks_n) * Tile::cta_m;
  const int tile_n = (blockIdx.x % blocks_n) * Tile::cta_n;
  const int warp_n = tile_n + warp * Tile::warp_n;
  int32_t accum[4][2][4] = {};
  for (int k_base = 0; k_base < Shape::gemm_k; k_base += Tile::stage_k) {
    generic_stage<Shape, Tile>(input, filter, shared, tile_m, tile_n, k_base,
                               thread, warp);
    __syncthreads();
    int8_t *a_base = shared;
    int8_t *b_base = a_base + Tile::cta_m * Tile::stage_k +
                     warp * Tile::warp_n * Tile::stage_k;
#pragma unroll
    for (int k = 0; k < Tile::stage_k; k += 32) {
      uint32_t a[4][4], b[2][2];
#pragma unroll
      for (int m = 0; m < 4; ++m)
        load_matrix_x4(a_base + generic_smem_offset<Tile::stage_k>(
                         m * 16 + lane % 16, k + (lane / 16) * 16),
                       a[m][0], a[m][1], a[m][2], a[m][3]);
#pragma unroll
      for (int n = 0; n < 2; ++n)
        load_matrix_x2(b_base + generic_smem_offset<Tile::stage_k>(
                         n * 8 + lane % 8, k + ((lane / 8) % 2) * 16),
                       b[n][0], b[n][1]);
#pragma unroll
      for (int m = 0; m < 4; ++m)
#pragma unroll
        for (int n = 0; n < 2; ++n)
          mma_m16n8k32(accum[m][n][0], accum[m][n][1], accum[m][n][2],
                        accum[m][n][3], a[m][0], a[m][1], a[m][2], a[m][3],
                        b[n][0], b[n][1]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int m = 0; m < 4; ++m)
#pragma unroll
    for (int n = 0; n < 2; ++n) {
      const int row = tile_m + m * 16 + lane / 4;
      const int col = warp_n + n * 8 + (lane & 3) * 2;
      // The paired column store is kept adjacent on the fast path so ptxas can
      // still fuse it into one 64-bit store; an odd gemm_n takes the tail.
      if (row < Shape::gemm_m) {
        if (col + 1 < Shape::gemm_n) {
          store_bias_pair<Shape::gemm_n>(output, bias, row, col, accum[m][n][0],
                                         accum[m][n][1]);
        } else if (col < Shape::gemm_n) {
          output[static_cast<size_t>(row) * Shape::gemm_n + col] =
              accum[m][n][0] + bias[col];
        }
      }
      if (row + 8 < Shape::gemm_m) {
        if (col + 1 < Shape::gemm_n) {
          store_bias_pair<Shape::gemm_n>(output, bias, row + 8, col,
                                         accum[m][n][2], accum[m][n][3]);
        } else if (col < Shape::gemm_n) {
          output[static_cast<size_t>(row + 8) * Shape::gemm_n + col] =
              accum[m][n][2] + bias[col];
        }
      }
    }
}

template <class Shape>
void launch_generic_sm80(const int8_t *input, const int8_t *filter,
                         const int32_t *bias, int32_t *output,
                         cudaStream_t stream = 0) {
  using Tile = GenericTile<>;
  constexpr int bm = (Shape::gemm_m + Tile::cta_m - 1) / Tile::cta_m;
  constexpr int bn = (Shape::gemm_n + Tile::cta_n - 1) / Tile::cta_n;
  generic_implicit_gemm_sm80_kernel<Shape, Tile>
      <<<bm * bn, Tile::threads, 0, stream>>>(input, filter, bias, output);
}
#endif
