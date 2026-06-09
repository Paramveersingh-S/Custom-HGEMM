#pragma once
#include <cuda_fp16.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

__device__ __forceinline__ void cp_async_cg_16(void* smem_ptr, const void* global_ptr) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(global_ptr)
        : "memory"
    );
}

__device__ __forceinline__ void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

#else

// Fallback for older architectures (like T4 sm_75)
__device__ __forceinline__ void cp_async_cg_16(void* smem_ptr, const void* global_ptr) {
    // 16 bytes = 1 float4
    *reinterpret_cast<float4*>(smem_ptr) = *reinterpret_cast<const float4*>(global_ptr);
}

__device__ __forceinline__ void cp_async_commit_group() {
    // No-op
}

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    // No-op
}

#endif
