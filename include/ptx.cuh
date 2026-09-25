#pragma once

#include <cstdint>

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
