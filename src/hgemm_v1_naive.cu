#include "hgemm_common.cuh"

// Phase 1: Naive GEMM (v1)
// Baseline performance, no shared memory or tensor cores.
// One thread computes one C[row][col] element.
__global__ void hgemm_v1_naive_kernel(const half* A, const half* B, half* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; k++) {
            // A is row-major (M, K) => A[row * K + k]
            // B is row-major (K, N) => B[k * N + col]
            float a_val = __half2float(A[row * K + k]);
            float b_val = __half2float(B[k * N + col]);
            acc += a_val * b_val;
        }
        C[row * N + col] = __float2half(acc);
    }
}

void launch_hgemm_v1_naive(const half* A, const half* B, half* C, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    hgemm_v1_naive_kernel<<<grid, block>>>(A, B, C, M, N, K);
    CHECK_CUDA(cudaGetLastError());
}
