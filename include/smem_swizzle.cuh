#pragma once
#include <cuda_fp16.h>
#include <mma.h>

// Swizzle pattern implementation to eliminate shared memory bank conflicts.
template<int B, int M, int S>
struct Swizzle {
    __device__ __forceinline__ static int apply(int row, int col) {
        int bit_b = (col >> (M + S)) & ((1 << B) - 1);
        int bit_m = (col >> S) & ((1 << M) - 1);
        int bit_s = col & ((1 << S) - 1);
        return (bit_b << (M + S)) | ((bit_m ^ (row & ((1 << M) - 1))) << S) | bit_s;
    }
};

using SwizzleA = Swizzle<3, 3, 3>; // For 128x32
using SwizzleB = Swizzle<3, 3, 3>; // For 32x128

// Helper to load into wmma fragments using ldmatrix
__device__ __forceinline__ void ldmatrix_x4(uint32_t* regs, const void* smem_ptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
        : "r"(smem_addr)
    );
#endif
}

__device__ __forceinline__ void ldmatrix_x4_notrans(uint32_t* regs, const void* smem_ptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
        : "r"(smem_addr)
    );
#endif
}
