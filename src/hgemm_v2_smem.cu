#include "hgemm_common.cuh"

// Phase 2: Shared Memory Tiling (v2)
// Template parameters for tile sizes
const int BM = 128;
const int BN = 128;
const int BK = 32;
const int TM = 8;
const int TN = 8;

__global__ void hgemm_v2_smem_kernel(const half* A, const half* B, half* C, int M, int N, int K) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x; // Thread x-index within block
    int ty = threadIdx.y; // Thread y-index within block
    int tid = ty * blockDim.x + tx; // Linear thread id (0 to 255)

    // A and B offsets for this block
    const half* A_block = A + by * BM * K;
    const half* B_block = B + bx * BN;
    half* C_block = C + by * BM * N + bx * BN;

    // Shared memory allocations
    // To avoid bank conflicts to some extent (though not fully swizzled), we could pad, 
    // but the spec just says naive SMEM tiling. We'll use unpadded to measure bank conflicts.
    __shared__ half s_A[BM * BK];
    __shared__ half s_B[BK * BN];

    // Thread local accumulator
    float accum[TM][TN] = {0.0f};

    // Calculate thread's output row and col within the 128x128 block
    // 256 threads. Let's arrange them logically as a 16x16 grid for computing.
    // threadIdx.y = 0..15, threadIdx.x = 0..15
    // Each thread computes a TMxTN (8x8) tile
    int row_start = (tid / 16) * TM;
    int col_start = (tid % 16) * TN;

    // Iterate over K dimension in chunks of BK
    for (int k = 0; k < K; k += BK) {
        // 1. Load A_tile[BM][BK] and B_tile[BK][BN] into SMEM
        // A is M x K. We need to load 128x32. Total 4096 elements.
        // 256 threads, so each thread loads 4096/256 = 16 elements.
        
        // Coalesced loading for A
        // A is row-major. To coalesce, threads should read adjacent elements in K.
        for (int i = 0; i < 16; i++) {
            int linear_idx = tid + i * 256;
            int a_row = linear_idx / BK;
            int a_col = linear_idx % BK;
            // Check bounds
            if ((by * BM + a_row) < M && (k + a_col) < K) {
                s_A[linear_idx] = A_block[a_row * K + k + a_col];
            } else {
                s_A[linear_idx] = __float2half(0.0f);
            }
        }

        // Coalesced loading for B
        // B is row-major. B is K x N. We need 32x128.
        for (int i = 0; i < 16; i++) {
            int linear_idx = tid + i * 256;
            int b_row = linear_idx / BN;
            int b_col = linear_idx % BN;
            // Check bounds
            if ((k + b_row) < K && (bx * BN + b_col) < N) {
                s_B[linear_idx] = B_block[(k + b_row) * N + b_col];
            } else {
                s_B[linear_idx] = __float2half(0.0f);
            }
        }

        __syncthreads();

        // 2. Compute
        for (int step = 0; step < BK; step++) {
            // Load TM elements of A and TN elements of B into registers
            float reg_A[TM];
            float reg_B[TN];

            for (int i = 0; i < TM; i++) {
                reg_A[i] = __half2float(s_A[(row_start + i) * BK + step]);
            }
            for (int j = 0; j < TN; j++) {
                reg_B[j] = __half2float(s_B[step * BN + col_start + j]);
            }

            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < TN; j++) {
                    accum[i][j] += reg_A[i] * reg_B[j];
                }
            }
        }

        __syncthreads();
    }

    // Write back to global memory C
    for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
            int g_row = by * BM + row_start + i;
            int g_col = bx * BN + col_start + j;
            if (g_row < M && g_col < N) {
                C[g_row * N + g_col] = __float2half(accum[i][j]);
            }
        }
    }
}

void launch_hgemm_v2_smem(const half* A, const half* B, half* C, int M, int N, int K) {
    // We configured 256 threads per block, represented as 16x16
    dim3 block(16, 16);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    hgemm_v2_smem_kernel<<<grid, block>>>(A, B, C, M, N, K);
    CHECK_CUDA(cudaGetLastError());
}
