#pragma once

#include "conv_shape.cuh"
#include "ptx.cuh"

#include <cuda_runtime.h>

// The strip family is intentionally a dispatch candidate, not a fallback.
// Its fixed geometry lets it reuse one 130x64 input strip across all three
// filter columns of a 128x64 CTA.
template <class Shape>
inline constexpr bool kSpatialStripEligible =
    Shape::r == 3 && Shape::s == 3 && Shape::stride_w == 1 &&
    Shape::dilation_w == 1 && Shape::c % 64 == 0 && Shape::k % 64 == 0 &&
    Shape::out_w % 128 == 0;

template <class Shape>
struct SpatialStripLayout {
  static constexpr int cta_m = 128;
  static constexpr int cta_n = 64;
  static constexpr int stage_k = 64;
  static constexpr int threads = 128;
  static constexpr int strip_pixels = cta_m + Shape::s - 1;
  static constexpr int strip_bytes = strip_pixels * stage_k;
  static constexpr int b_bytes = cta_n * stage_k;
  static constexpr int shared_bytes = strip_bytes + b_bytes;
  static constexpr int strip_vectors = strip_bytes / 16;
  static constexpr int b_vectors_per_thread = b_bytes / 16 / threads;
};

__device__ __forceinline__ int strip_swizzle(int byte_offset) {
  return byte_offset ^ ((byte_offset & 0x1c0) >> 2);
}

template <class Shape>
__device__ __forceinline__ int4 strip_load_input(const int8_t *input, int batch,
                                                   int y, int x, int channel) {
  if (y < 0 || y >= Shape::h || x < 0 || x >= Shape::w) {
    return make_int4(0, 0, 0, 0);
  }
  return *reinterpret_cast<const int4 *>(
      input + ((batch * Shape::h + y) * Shape::w + x) * Shape::c + channel);
}

template <class Shape>
__device__ __forceinline__ void stage_strip_a(
    int8_t *shared_a, const int8_t *input, int batch, int output_y,
    int output_x, int channel_group, int filter_y, int thread) {
  using L = SpatialStripLayout<Shape>;
  constexpr int vectors_per_pixel = L::stage_k / 16;
  for (int vector = thread; vector < L::strip_vectors; vector += L::threads) {
    const int strip_x = vector / vectors_per_pixel;
    const int channel = channel_group * L::stage_k +
                        (vector % vectors_per_pixel) * 16;
    const int4 value = strip_load_input<Shape>(
        input, batch, output_y * Shape::stride_h - Shape::pad_h +
                          filter_y * Shape::dilation_h,
        output_x - Shape::pad_w + strip_x, channel);
    *reinterpret_cast<int4 *>(shared_a + strip_swizzle(vector * 16)) = value;
  }
}

template <class Shape>
struct StripBPrefetch {
  int4 values[SpatialStripLayout<Shape>::b_vectors_per_thread];
};

template <class Shape>
__device__ __forceinline__ void prefetch_strip_b(
    StripBPrefetch<Shape> &prefetch, const int8_t *filter, int n_base,
    int channel_group, int filter_y, int filter_x, int thread) {
  using L = SpatialStripLayout<Shape>;
  constexpr int vectors_per_row = L::stage_k / 16;
#pragma unroll
  for (int i = 0; i < L::b_vectors_per_thread; ++i) {
    const int vector = thread + i * L::threads;
    const int output_channel = vector / vectors_per_row;
    const int channel = channel_group * L::stage_k +
                        (vector % vectors_per_row) * 16;
    prefetch.values[i] = *reinterpret_cast<const int4 *>(
        filter + (((n_base + output_channel) * Shape::r + filter_y) * Shape::s +
                  filter_x) * Shape::c + channel);
  }
}

template <class Shape>
__device__ __forceinline__ void store_strip_b(
    int8_t *shared_b, const StripBPrefetch<Shape> &prefetch, int thread) {
  using L = SpatialStripLayout<Shape>;
#pragma unroll
  for (int i = 0; i < L::b_vectors_per_thread; ++i) {
    const int vector = thread + i * L::threads;
    *reinterpret_cast<int4 *>(shared_b + strip_swizzle(vector * 16)) =
        prefetch.values[i];
  }
}

template <class Shape>
__device__ __forceinline__ void strip_load_fragments(
    uint32_t (&a)[8], uint32_t (&b)[4], const int8_t *shared_a,
    const int8_t *shared_b, int warp_m_group, int warp_n_group, int lane,
    int filter_x, int k_step) {
  using L = SpatialStripLayout<Shape>;
  const int a_base = (filter_x + warp_m_group + (lane & 7)) * L::stage_k;
#pragma unroll
  for (int group = 0; group < 2; ++group) {
    const int source_m8 = group * 4 + (lane >> 3);
    load_matrix_x4(
        shared_a + strip_swizzle(a_base + source_m8 * 16 * L::stage_k + k_step),
        a[group * 4], a[group * 4 + 1], a[group * 4 + 2], a[group * 4 + 3]);
  }
  const int b_base = (warp_n_group + (lane & 7)) * L::stage_k;
  load_matrix_x4(
      shared_b + strip_swizzle(b_base + (lane >> 3) * 16 * L::stage_k + k_step),
      b[0], b[1], b[2], b[3]);
}

template <class Shape>
__global__ __launch_bounds__(128, 1) void spatial_strip_kernel(
    const int8_t *__restrict__ input, const int8_t *__restrict__ filter,
    const int32_t *__restrict__ bias, int32_t *__restrict__ output) {
  using L = SpatialStripLayout<Shape>;
  __shared__ __align__(16) int8_t shared[L::shared_bytes];
  int8_t *shared_a = shared;
  int8_t *shared_b = shared + L::strip_bytes;
  const int thread = threadIdx.x;
  const int lane = thread & 31;
  const int warp = thread >> 5;
  constexpr int blocks_n = Shape::gemm_n / L::cta_n;
  const int block_m = (blockIdx.x / blocks_n) * L::cta_m;
  const int n_base = (blockIdx.x % blocks_n) * L::cta_n;
  const int pixel = block_m % (Shape::out_h * Shape::out_w);
  const int batch = block_m / (Shape::out_h * Shape::out_w);
  const int output_y = pixel / Shape::out_w;
  const int output_x = pixel % Shape::out_w;
  const int warp_m_group = (warp >> 1) << 3;
  const int warp_n_group = (warp & 1) << 3;
  int32_t accumulators[8][4][2] = {};

#pragma unroll 1
  for (int channel_group = 0; channel_group < Shape::c / L::stage_k;
       ++channel_group) {
#pragma unroll 1
    for (int filter_y = 0; filter_y < Shape::r; ++filter_y) {
      stage_strip_a<Shape>(shared_a, input, batch, output_y, output_x,
                           channel_group, filter_y, thread);
      __syncthreads();
      StripBPrefetch<Shape> prefetched;
      prefetch_strip_b<Shape>(prefetched, filter, n_base, channel_group,
                              filter_y, 0, thread);
      store_strip_b<Shape>(shared_b, prefetched, thread);
      __syncthreads();

#pragma unroll
      for (int filter_x = 0; filter_x < Shape::s; ++filter_x) {
        const bool has_next = filter_x + 1 < Shape::s;
        if (has_next) {
          prefetch_strip_b<Shape>(prefetched, filter, n_base, channel_group,
                                  filter_y, filter_x + 1, thread);
        }
#pragma unroll
        for (int k_step = 0; k_step < L::stage_k; k_step += 16) {
          uint32_t a[8], b[4];
          strip_load_fragments<Shape>(a, b, shared_a, shared_b, warp_m_group,
                                      warp_n_group, lane, filter_x, k_step);
#pragma unroll
          for (int mma_m = 0; mma_m < 8; ++mma_m) {
#pragma unroll
            for (int mma_n = 0; mma_n < 4; ++mma_n) {
              mma_m8n8k16(accumulators[mma_m][mma_n][0],
                          accumulators[mma_m][mma_n][1], a[mma_m], b[mma_n]);
            }
          }
        }
        __syncthreads();
        if (has_next) {
          store_strip_b<Shape>(shared_b, prefetched, thread);
          __syncthreads();
        }
      }
    }
  }

#pragma unroll
  for (int mma_m = 0; mma_m < 8; ++mma_m) {
#pragma unroll
    for (int mma_n = 0; mma_n < 4; ++mma_n) {
      const int row = block_m + warp_m_group + lane / 4 + mma_m * 16;
      const int col = n_base + warp_n_group + (lane & 3) * 2 + mma_n * 16;
      output[row * Shape::gemm_n + col] =
          accumulators[mma_m][mma_n][0] + bias[col];
      output[row * Shape::gemm_n + col + 1] =
          accumulators[mma_m][mma_n][1] + bias[col + 1];
    }
  }
}

template <class Shape>
void launch_spatial_strip_sm75(const int8_t *input, const int8_t *filter,
                               const int32_t *bias, int32_t *output,
                               cudaStream_t stream = 0) {
  static_assert(kSpatialStripEligible<Shape>,
                "shape is not eligible for spatial strip");
  constexpr int blocks_m = Shape::gemm_m / SpatialStripLayout<Shape>::cta_m;
  constexpr int blocks_n = Shape::gemm_n / SpatialStripLayout<Shape>::cta_n;
  spatial_strip_kernel<Shape><<<blocks_m * blocks_n, 128, 0, stream>>>(
      input, filter, bias, output);
}

#if INT8_LAB_MMA_SM80
// CGIR's SM80 spatial-strip path: m16n8k32 uses four A, two B, and four C
// registers.  The lane/K mapping is intentionally identical to
// TensorCoreMMAInstruction::getFrag{A,B,C}ElementSubscript.
template <class Shape>
__global__ __launch_bounds__(128, 1) void spatial_strip_sm80_kernel(
    const int8_t *__restrict__ input, const int8_t *__restrict__ filter,
    const int32_t *__restrict__ bias, int32_t *__restrict__ output) {
  using L = SpatialStripLayout<Shape>;
  __shared__ __align__(16) int8_t shared[L::shared_bytes];
  int8_t *shared_a = shared, *shared_b = shared + L::strip_bytes;
  const int thread = threadIdx.x, lane = thread & 31, warp = thread >> 5;
  constexpr int blocks_n = Shape::gemm_n / L::cta_n;
  const int block_m = (blockIdx.x / blocks_n) * L::cta_m;
  const int n_base = (blockIdx.x % blocks_n) * L::cta_n;
  const int pixel = block_m % (Shape::out_h * Shape::out_w);
  const int batch = block_m / (Shape::out_h * Shape::out_w);
  const int oy = pixel / Shape::out_w, ox = pixel % Shape::out_w;
  const int warp_m = (warp >> 1) * 64, warp_n = (warp & 1) * 32;
  int32_t acc[4][4][4] = {};
#pragma unroll 1
  for (int cg = 0; cg < Shape::c / 64; ++cg) {
#pragma unroll 1
    for (int fy = 0; fy < Shape::r; ++fy) {
      stage_strip_a<Shape>(shared_a, input, batch, oy, ox, cg, fy, thread);
      __syncthreads();
      StripBPrefetch<Shape> prefetched;
      prefetch_strip_b<Shape>(prefetched, filter, n_base, cg, fy, 0, thread);
      store_strip_b<Shape>(shared_b, prefetched, thread);
      __syncthreads();
#pragma unroll
      for (int fx = 0; fx < Shape::s; ++fx) {
        const bool next = fx + 1 < Shape::s;
        if (next) prefetch_strip_b<Shape>(prefetched, filter, n_base, cg, fy, fx + 1, thread);
        uint32_t a[2][4][4], b[2][4][2];
        auto load = [&](int buffer, int k) {
#pragma unroll
          for (int m = 0; m < 4; ++m)
            load_matrix_x4(shared_a + strip_swizzle(
                (fx + warp_m + m * 16 + lane % 16) * 64 + k + (lane / 16) * 16),
                a[buffer][m][0], a[buffer][m][1], a[buffer][m][2], a[buffer][m][3]);
#pragma unroll
          for (int n = 0; n < 4; ++n)
            load_matrix_x2(shared_b + strip_swizzle(
                (warp_n + n * 8 + lane % 8) * 64 + k + ((lane / 8) % 2) * 16),
                b[buffer][n][0], b[buffer][n][1]);
        };
        load(0, 0);
#pragma unroll
        for (int k = 0; k < 64; k += 32) {
          const int current = (k / 32) & 1;
          if (k + 32 < 64) load(current ^ 1, k + 32);
#pragma unroll
          for (int m = 0; m < 4; ++m)
#pragma unroll
            for (int n = 0; n < 4; ++n)
              mma_m16n8k32(acc[m][n][0], acc[m][n][1], acc[m][n][2], acc[m][n][3],
                            a[current][m][0], a[current][m][1], a[current][m][2], a[current][m][3],
                            b[current][n][0], b[current][n][1]);
        }
        __syncthreads();
        if (next) { store_strip_b<Shape>(shared_b, prefetched, thread); __syncthreads(); }
      }
    }
  }
#pragma unroll
  for (int m = 0; m < 4; ++m)
#pragma unroll
    for (int n = 0; n < 4; ++n) {
      const int row = block_m + warp_m + m * 16 + lane / 4;
      const int col = n_base + warp_n + n * 8 + (lane & 3) * 2;
      output[row * Shape::gemm_n + col] = acc[m][n][0] + bias[col];
      output[row * Shape::gemm_n + col + 1] = acc[m][n][1] + bias[col + 1];
      output[(row + 8) * Shape::gemm_n + col] = acc[m][n][2] + bias[col];
      output[(row + 8) * Shape::gemm_n + col + 1] = acc[m][n][3] + bias[col + 1];
    }
}
#endif

template <class Shape>
void launch_spatial_strip(const int8_t *input, const int8_t *filter,
                          const int32_t *bias, int32_t *output,
                          cudaStream_t stream = 0) {
  static_assert(kSpatialStripEligible<Shape>);
#if INT8_LAB_MMA_SM80
  constexpr int bm = Shape::gemm_m / SpatialStripLayout<Shape>::cta_m;
  constexpr int bn = Shape::gemm_n / SpatialStripLayout<Shape>::cta_n;
  spatial_strip_sm80_kernel<Shape><<<bm * bn, 128, 0, stream>>>(input, filter, bias, output);
#else
  launch_spatial_strip_sm75<Shape>(input, filter, bias, output, stream);
#endif
}
