#pragma once

#include <cstdint>

// Minimum CTAs per SM requested from ptxas.  Both spatial kernels carry 64
// INT32 accumulators per thread and default to ~1.5x the register budget that
// four concurrent CTAs allow, so without this they run at one third of the
// occupancy they could.  Measured on sm_120/big_3x3_s1: strip 9.34 -> 8.52 ms,
// halo 10.10 -> 7.41 ms.  Five spills and costs roughly 2x, so this is a
// per-architecture tuning knob, not a constant of nature.
// m8n8k16 needs one more register per accumulator pair and spills at four.
#ifndef INT8_LAB_TARGET_CTAS_PER_SM
#define INT8_LAB_TARGET_CTAS_PER_SM 4
#endif
#ifndef INT8_LAB_MIN_CTAS_PER_SM
#if INT8_LAB_MMA_SM80
#define INT8_LAB_MIN_CTAS_PER_SM INT8_LAB_TARGET_CTAS_PER_SM
#else
// m8n8k16 needs one more register per accumulator pair and spills a step
// earlier, so it targets one fewer CTA than the machine description asks for.
#define INT8_LAB_MIN_CTAS_PER_SM \
  (INT8_LAB_TARGET_CTAS_PER_SM > 1 ? INT8_LAB_TARGET_CTAS_PER_SM - 1 : 1)
#endif
#endif

__device__ __forceinline__ void load_matrix_x2(const void *source,
                                                uint32_t &r0, uint32_t &r1) {
  uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(source));
  asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0,%1},[%2];"
               : "=r"(r0), "=r"(r1) : "r"(address));
}

__device__ __forceinline__ void load_matrix_x4(const void *source,
                                                uint32_t &r0, uint32_t &r1,
                                                uint32_t &r2, uint32_t &r3) {
  uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(source));
  asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3},[%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(address));
}

__device__ __forceinline__ void mma_m8n8k16(int32_t &d0, int32_t &d1,
                                             uint32_t a, uint32_t b) {
  int32_t next0, next1;
  asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
               "{%0,%1},{%2},{%3},{%4,%5};"
               : "=r"(next0), "=r"(next1)
               : "r"(a), "r"(b), "r"(d0), "r"(d1));
  d0 = next0;
  d1 = next1;
}

__device__ __forceinline__ void mma_m16n8k32(
    int32_t &d0, int32_t &d1, int32_t &d2, int32_t &d3, uint32_t a0,
    uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  int32_t next0, next1, next2, next3;
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
               "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
               : "=r"(next0), "=r"(next1), "=r"(next2), "=r"(next3)
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                 "r"(d0), "r"(d1), "r"(d2), "r"(d3));
  d0 = next0;
  d1 = next1;
  d2 = next2;
  d3 = next3;
}

// The INT8 MMA C fragments hand each lane two adjacent output columns, but
// ptxas does not fuse the two scalar stores on its own -- it emits 2x STG.E,
// which touches 32 bytes of sectors per 16 useful bytes (2x write
// amplification).  Emitting the pair explicitly gives STG.E.64.  `col` is
// always even by construction; an odd row pitch would break the 8-byte
// alignment, so that case keeps the scalar stores.
template <int Pitch>
__device__ __forceinline__ void store_bias_pair(int32_t *__restrict__ output,
                                                const int32_t *__restrict__ bias,
                                                int row, int col, int32_t v0,
                                                int32_t v1) {
  int32_t *destination = output + static_cast<size_t>(row) * Pitch + col;
  if constexpr (Pitch % 2 == 0) {
    *reinterpret_cast<int2 *>(destination) =
        make_int2(v0 + bias[col], v1 + bias[col + 1]);
  } else {
    destination[0] = v0 + bias[col];
    destination[1] = v1 + bias[col + 1];
  }
}
