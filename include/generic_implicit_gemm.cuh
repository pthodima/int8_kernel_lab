#pragma once

#include "conv_shape.cuh"
#include "ptx.cuh"

#include <cuda_runtime.h>

// A staged implicit-GEMM fallback, covering every shape with C % 16 == 0.
//
// The CTA tile, the warp tile and the staging depth are all template
// parameters, and the kernels derive their fragment counts, loop bounds and
// accumulator extents from them.  An earlier version fixed the shape at
// 64x64x32 with 64x16 warp tiles: `GenericTile` templated only `StageK`, both
// launchers instantiated the default, and the kernel bodies hardcoded
// `accum[8][2][2]` and `for (m < 8)` rather than deriving them, so the struct
// only looked parameterised.
//
// `warps` is derived rather than free: the warp grid is exactly
// (cta_m / warp_m) x (cta_n / warp_n), so the warp count follows from the two
// tile shapes and cannot be chosen independently of them.
//
// Per-thread accumulator registers are warp_m * warp_n / 32 under both MMA
// families, which is the first thing to check when picking a warp tile: a
// 64x64 warp tile costs 128 registers of accumulator alone.
template <int CtaM = 64, int CtaN = 64, int WarpM = 64, int WarpN = 16,
          int StageK = 32>
struct GenericTile {
  static constexpr int cta_m = CtaM;
  static constexpr int cta_n = CtaN;
  static constexpr int warp_m = WarpM;
  static constexpr int warp_n = WarpN;
  static constexpr int warps_m = CtaM / WarpM;
  static constexpr int warps_n = CtaN / WarpN;
  static constexpr int warps = warps_m * warps_n;
  static constexpr int threads = warps * 32;
  static constexpr int stage_k = StageK;
  static constexpr int mma_m = 8;
  static constexpr int mma_n = 8;
  static constexpr int mma_k = 16;
  static constexpr int shared_bytes = (CtaM + CtaN) * StageK;
  static constexpr int accumulator_registers = WarpM * WarpN / 32;

  static_assert(CtaM % WarpM == 0, "cta_m must be a whole number of warp tiles");
  static_assert(CtaN % WarpN == 0, "cta_n must be a whole number of warp tiles");
  // ldmatrix returns two 8-row matrices per x2 call, and m16n8k32 consumes
  // sixteen A rows per fragment, so both families want warp_m % 16.
  static_assert(WarpM % 16 == 0, "warp_m must be a multiple of 16");
  static_assert(WarpN % 8 == 0, "warp_n must be a multiple of 8");
  static_assert(StageK % 16 == 0, "stage_k must be a multiple of the MMA K");
  static_assert(threads >= 32 && threads <= 1024, "1 to 32 warps per CTA");
  static_assert(shared_bytes <= 48 * 1024,
                "staged tile must fit the static shared-memory allocation");
};

#ifndef INT8_LAB_GENERIC_CTA_M
#define INT8_LAB_GENERIC_CTA_M 64
#endif
#ifndef INT8_LAB_GENERIC_CTA_N
#define INT8_LAB_GENERIC_CTA_N 64
#endif
#ifndef INT8_LAB_GENERIC_WARP_M
#define INT8_LAB_GENERIC_WARP_M 64
#endif
#ifndef INT8_LAB_GENERIC_WARP_N
#define INT8_LAB_GENERIC_WARP_N 16
#endif
#ifndef INT8_LAB_GENERIC_STAGE_K
#define INT8_LAB_GENERIC_STAGE_K 32
#endif
using DefaultGenericTile =
    GenericTile<INT8_LAB_GENERIC_CTA_M, INT8_LAB_GENERIC_CTA_N,
                INT8_LAB_GENERIC_WARP_M, INT8_LAB_GENERIC_WARP_N,
                INT8_LAB_GENERIC_STAGE_K>;

// XOR swizzle over an 8-row group, stored vector-major: eight rows at one
// vector land on eight consecutive 16-byte slots, which is 128 bytes and
// therefore every bank exactly once, so an ldmatrix fragment is conflict-free
// whatever row it starts on.  The rotation keeps the staging store, whose
// lanes walk vectors rather than rows, spread over the same eight bank quads.
//
// This was verified directly rather than argued: an isolated kernel doing
// nothing but this store reports zero conflicts at every thread count, both
// stage_k values, one slab or two, and with ldmatrix interleaved.
//
// Do not trust l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st on the
// real kernel here.  It reports 0-19% across tile shapes, but that counter
// measures *excess wavefronts*, and a store whose data comes from a global
// load that has not returned is replayed into the same counter.  Feeding the
// identical store addresses a constant instead of a global load drops it from
// 2283 conflicts / 29931 wavefronts to 0 / 27648, the ideal count; replacing
// the swizzle with a linear layout leaves it unchanged; and the figure is not
// reproducible run to run.  The apparent correlation with warp count is
// outstanding global loads, not tile geometry.
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

// Stages one cta_m x stage_k slab of A and one cta_n x stage_k slab of B.
// Both are staged by the whole CTA rather than per warp: with a 2D warp grid
// several warps share a column range, and a per-warp B slab would store it
// more than once.
template <class Shape, class Tile>
__device__ __forceinline__ void generic_stage(
    const int8_t *input, const int8_t *filter, int8_t *shared, int tile_m,
    int tile_n, int k_base, int thread) {
  constexpr int vectors_per_row = Tile::stage_k / 16;
  constexpr int a_vectors = Tile::cta_m * vectors_per_row;
  constexpr int b_vectors = Tile::cta_n * vectors_per_row;
  int8_t *shared_a = shared;
  int8_t *shared_b = shared_a + Tile::cta_m * Tile::stage_k;

  for (int vector = thread; vector < a_vectors; vector += Tile::threads) {
    const int row = vector / vectors_per_row;
    const int inner = k_base + (vector % vectors_per_row) * 16;
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

  for (int vector = thread; vector < b_vectors; vector += Tile::threads) {
    const int row = vector / vectors_per_row;
    const int inner = k_base + (vector % vectors_per_row) * 16;
    const int n = tile_n + row;
    uint4 value = make_uint4(0, 0, 0, 0);
    if (n < Shape::gemm_n && inner < Shape::gemm_k) {
      const int channel = inner % Shape::c;
      const int fx = (inner / Shape::c) % Shape::s;
      const int fy = (inner / Shape::c / Shape::s) % Shape::r;
      value = *reinterpret_cast<const uint4 *>(
          filter + ((n * Shape::r + fy) * Shape::s + fx) * Shape::c + channel);
    }
    *reinterpret_cast<uint4 *>(
        shared_b + generic_smem_offset<Tile::stage_k>(row, inner - k_base)) =
        value;
  }
}

// Writes one accumulator pair, taking the paired 64-bit store whenever the
// column tail allows it.
template <class Shape>
__device__ __forceinline__ void generic_store(int32_t *__restrict__ output,
                                              const int32_t *__restrict__ bias,
                                              int row, int col, int32_t v0,
                                              int32_t v1) {
  if (row >= Shape::gemm_m) return;
  if (col + 1 < Shape::gemm_n) {
    store_bias_pair<Shape::gemm_n>(output, bias, row, col, v0, v1);
  } else if (col < Shape::gemm_n) {
    output[static_cast<size_t>(row) * Shape::gemm_n + col] = v0 + bias[col];
  }
}

template <class Shape, class Tile = DefaultGenericTile>
__global__ __launch_bounds__(Tile::threads, 1) void generic_implicit_gemm_kernel(
    const int8_t *__restrict__ input, const int8_t *__restrict__ filter,
    const int32_t *__restrict__ bias, int32_t *__restrict__ output) {
  // m8n8k16 reads both operands through ldmatrix.x2, which returns two 8-row
  // matrices per call, so the warp's N extent also comes in pairs of tiles.
  static_assert(Tile::warp_n % 16 == 0,
                "m8n8k16 stages B in pairs of 8-column tiles");
  constexpr int kAccM = Tile::warp_m / Tile::mma_m;
  constexpr int kAccN = Tile::warp_n / Tile::mma_n;

  __shared__ __align__(16) int8_t shared[Tile::shared_bytes];
  const int thread = threadIdx.x;
  const int warp = thread >> 5;
  const int lane = thread & 31;
  const int warp_m_base = (warp / Tile::warps_n) * Tile::warp_m;
  const int warp_n_base = (warp % Tile::warps_n) * Tile::warp_n;
  const int blocks_n = (Shape::gemm_n + Tile::cta_n - 1) / Tile::cta_n;
  const int tile_m = (blockIdx.x / blocks_n) * Tile::cta_m;
  const int tile_n = (blockIdx.x % blocks_n) * Tile::cta_n;
  int32_t accum[kAccM][kAccN][2] = {};

  for (int k_base = 0; k_base < Shape::gemm_k; k_base += Tile::stage_k) {
    generic_stage<Shape, Tile>(input, filter, shared, tile_m, tile_n, k_base,
                               thread);
    __syncthreads();
    int8_t *shared_a = shared;
    int8_t *shared_b = shared_a + Tile::cta_m * Tile::stage_k;
#pragma unroll
    for (int k = 0; k < Tile::stage_k; k += Tile::mma_k) {
      uint32_t a[kAccM];
      uint32_t b[kAccN];
#pragma unroll
      for (int m = 0; m < kAccM; m += 2) {
        load_matrix_x2(shared_a + generic_smem_offset<Tile::stage_k>(
                                      warp_m_base + m * Tile::mma_m + lane % 16,
                                      k),
                       a[m], a[m + 1]);
      }
#pragma unroll
      for (int n = 0; n < kAccN; n += 2) {
        load_matrix_x2(shared_b + generic_smem_offset<Tile::stage_k>(
                                      warp_n_base + n * Tile::mma_n + lane % 16,
                                      k),
                       b[n], b[n + 1]);
      }
#pragma unroll
      for (int m = 0; m < kAccM; ++m) {
#pragma unroll
        for (int n = 0; n < kAccN; ++n) {
          mma_m8n8k16(accum[m][n][0], accum[m][n][1], a[m], b[n]);
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int m = 0; m < kAccM; ++m) {
#pragma unroll
    for (int n = 0; n < kAccN; ++n) {
      const int row = tile_m + warp_m_base + m * Tile::mma_m + lane / 4;
      const int col = tile_n + warp_n_base + n * Tile::mma_n + (lane & 3) * 2;
      generic_store<Shape>(output, bias, row, col, accum[m][n][0],
                           accum[m][n][1]);
    }
  }
}

template <class Shape, class Tile = DefaultGenericTile>
void launch_generic(const int8_t *input, const int8_t *filter,
                    const int32_t *bias, int32_t *output,
                    cudaStream_t stream = 0) {
  constexpr int blocks_m = (Shape::gemm_m + Tile::cta_m - 1) / Tile::cta_m;
  constexpr int blocks_n = (Shape::gemm_n + Tile::cta_n - 1) / Tile::cta_n;
  generic_implicit_gemm_kernel<Shape, Tile>
      <<<blocks_m * blocks_n, Tile::threads, 0, stream>>>(input, filter, bias,
                                                          output);
}

#if INT8_LAB_MMA_SM80
// This is the CGIR SM80 fragment contract: A has four U32 registers, B has
// two, C has four; lane subscript K offsets are 0 or 16.
template <class Shape, class Tile = DefaultGenericTile>
__global__ __launch_bounds__(Tile::threads, 1) void generic_implicit_gemm_sm80_kernel(
    const int8_t *__restrict__ input, const int8_t *__restrict__ filter,
    const int32_t *__restrict__ bias, int32_t *__restrict__ output) {
  static_assert(Tile::stage_k % 32 == 0, "m16n8k32 consumes 32 channels a step");
  constexpr int kAccM = Tile::warp_m / 16;
  constexpr int kAccN = Tile::warp_n / 8;

  __shared__ __align__(16) int8_t shared[Tile::shared_bytes];
  const int thread = threadIdx.x, warp = thread >> 5, lane = thread & 31;
  const int warp_m_base = (warp / Tile::warps_n) * Tile::warp_m;
  const int warp_n_base = (warp % Tile::warps_n) * Tile::warp_n;
  const int blocks_n = (Shape::gemm_n + Tile::cta_n - 1) / Tile::cta_n;
  const int tile_m = (blockIdx.x / blocks_n) * Tile::cta_m;
  const int tile_n = (blockIdx.x % blocks_n) * Tile::cta_n;
  int32_t accum[kAccM][kAccN][4] = {};

  for (int k_base = 0; k_base < Shape::gemm_k; k_base += Tile::stage_k) {
    generic_stage<Shape, Tile>(input, filter, shared, tile_m, tile_n, k_base,
                               thread);
    __syncthreads();
    int8_t *a_base = shared;
    int8_t *b_base = a_base + Tile::cta_m * Tile::stage_k;
#pragma unroll
    for (int k = 0; k < Tile::stage_k; k += 32) {
      uint32_t a[kAccM][4], b[kAccN][2];
#pragma unroll
      for (int m = 0; m < kAccM; ++m)
        load_matrix_x4(a_base + generic_smem_offset<Tile::stage_k>(
                                    warp_m_base + m * 16 + lane % 16,
                                    k + (lane / 16) * 16),
                       a[m][0], a[m][1], a[m][2], a[m][3]);
#pragma unroll
      for (int n = 0; n < kAccN; ++n)
        load_matrix_x2(b_base + generic_smem_offset<Tile::stage_k>(
                                    warp_n_base + n * 8 + lane % 8,
                                    k + ((lane / 8) % 2) * 16),
                       b[n][0], b[n][1]);
#pragma unroll
      for (int m = 0; m < kAccM; ++m)
#pragma unroll
        for (int n = 0; n < kAccN; ++n)
          mma_m16n8k32(accum[m][n][0], accum[m][n][1], accum[m][n][2],
                        accum[m][n][3], a[m][0], a[m][1], a[m][2], a[m][3],
                        b[n][0], b[n][1]);
    }
    __syncthreads();
  }

#pragma unroll
  for (int m = 0; m < kAccM; ++m)
#pragma unroll
    for (int n = 0; n < kAccN; ++n) {
      const int row = tile_m + warp_m_base + m * 16 + lane / 4;
      const int col = tile_n + warp_n_base + n * 8 + (lane & 3) * 2;
      generic_store<Shape>(output, bias, row, col, accum[m][n][0],
                           accum[m][n][1]);
      generic_store<Shape>(output, bias, row + 8, col, accum[m][n][2],
                           accum[m][n][3]);
    }
}

template <class Shape, class Tile = DefaultGenericTile>
void launch_generic_sm80(const int8_t *input, const int8_t *filter,
                         const int32_t *bias, int32_t *output,
                         cudaStream_t stream = 0) {
  constexpr int bm = (Shape::gemm_m + Tile::cta_m - 1) / Tile::cta_m;
  constexpr int bn = (Shape::gemm_n + Tile::cta_n - 1) / Tile::cta_n;
  generic_implicit_gemm_sm80_kernel<Shape, Tile>
      <<<bm * bn, Tile::threads, 0, stream>>>(input, filter, bias, output);
}
#endif
