#include "hgemm_common.cuh"
#include <mma.h>

using namespace nvcuda;

// Phase 3: WMMA Tensor Core API (v3)
const int BM = 128;
const int BN = 128;
const int BK = 32;

const int WM = 32;
const int WN = 64;

__global__ void hgemm_v3_wmma_kernel(const half* A, const half* B, half* C, int M, int N, int K) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int warpId = tid / 32;
    int laneId = tid % 32;

    int warp_m = warpId % 4;
    int warp_n = warpId / 4;

    const half* A_block = A + by * BM * K;
    const half* B_block = B + bx * BN;
    
    __shared__ half s_A[BK][BM];
    __shared__ half s_B[BK][BN];
    
    // For storing the float accumulator to cast to half
    // We do it tile by tile to save shared memory. 
    // 8 warps, each stores a 16x16 float tile at a time.
    __shared__ float s_C_tile[8][16][16];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag[2]; 
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[4]; 
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[2][4];

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }

    for (int k = 0; k < K; k += BK) {
        // Load A into s_A (transposed: s_A[k][m] = A[m][k])
        for (int i = 0; i < 16; i++) {
            int linear_idx = tid + i * 256;
            int a_row = linear_idx / BK;
            int a_col = linear_idx % BK;
            
            if ((by * BM + a_row) < M && (k + a_col) < K) {
                s_A[a_col][a_row] = A_block[a_row * K + k + a_col];
            } else {
                s_A[a_col][a_row] = __float2half(0.0f);
            }
        }

        // Load B into s_B (s_B[k][n] = B[k][n])
        for (int i = 0; i < 16; i++) {
            int linear_idx = tid + i * 256;
            int b_row = linear_idx / BN;
            int b_col = linear_idx % BN;
            
            if ((k + b_row) < K && (bx * BN + b_col) < N) {
                s_B[b_row][b_col] = B_block[(k + b_row) * N + b_col];
            } else {
                s_B[b_row][b_col] = __float2half(0.0f);
            }
        }

        __syncthreads();

        for (int step = 0; step < BK; step += 16) {
            for (int i = 0; i < 2; i++) {
                int smem_m = warp_m * WM + i * 16;
                wmma::load_matrix_sync(a_frag[i], &s_A[step][smem_m], BM);
            }

            for (int j = 0; j < 4; j++) {
                int smem_n = warp_n * WN + j * 16;
                wmma::load_matrix_sync(b_frag[j], &s_B[step][smem_n], BN);
            }

            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 4; j++) {
                    wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
                }
            }
        }
        __syncthreads();
    }

    // Store c_frag to global memory
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
            int g_row = by * BM + warp_m * WM + i * 16;
            int g_col = bx * BN + warp_n * WN + j * 16;
            
            // Store the 16x16 float fragment to shared memory
            wmma::store_matrix_sync(&s_C_tile[warpId][0][0], c_frag[i][j], 16, wmma::mem_row_major);
            
            // Each of the 32 threads in the warp copies 8 elements from s_C_tile to global C
            for (int t = 0; t < 8; t++) {
                int elem_idx = laneId + t * 32; // 0 to 255
                int tile_row = elem_idx / 16;
                int tile_col = elem_idx % 16;
                
                if (g_row + tile_row < M && g_col + tile_col < N) {
                    C[(g_row + tile_row) * N + (g_col + tile_col)] = __float2half(s_C_tile[warpId][tile_row][tile_col]);
                }
            }
        }
    }
}

void launch_hgemm_v3_wmma(const half* A, const half* B, half* C, int M, int N, int K) {
    dim3 block(256);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    hgemm_v3_wmma_kernel<<<grid, block>>>(A, B, C, M, N, K);
    CHECK_CUDA(cudaGetLastError());
}
