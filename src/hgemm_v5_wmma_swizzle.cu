#include "hgemm_common.cuh"
#include "pipeline.cuh"
#include <mma.h>

using namespace nvcuda;

// Phase 5: SMEM Swizzling (v5)
const int BM = 128;
const int BN = 128;
const int BK = 32;
const int WM = 32;
const int WN = 64;

__global__ void hgemm_v5_wmma_swizzle_kernel(const half* A, const half* B, half* C, int M, int N, int K) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int warpId = tid / 32;
    int laneId = tid % 32;

    int warp_m = warpId % 4;
    int warp_n = warpId / 4;

    const half* A_block = A + by * BM * K;
    const half* B_block = B + bx * BN;
    
    __shared__ half s_A[2][BM][BK + 8]; // Padded to resolve bank conflicts!
    __shared__ half s_B[2][BK][BN + 8];
    __shared__ float s_C_tile[8][16][16];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[2]; 
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[4]; 
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[2][4];

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }

    int num_k_tiles = K / BK;

    for (int i = 0; i < 2; i++) {
        int linear_idx = (tid + i * 256) * 8; 
        int a_row = linear_idx / BK;
        int a_col = linear_idx % BK;
        
        if ((by * BM + a_row) < M && a_col < K) {
            cp_async_cg_16(&s_A[0][a_row][a_col], &A_block[a_row * K + a_col]);
        } else {
            float4 zeros = {0.0f, 0.0f, 0.0f, 0.0f};
            *reinterpret_cast<float4*>(&s_A[0][a_row][a_col]) = zeros;
        }
    }

    for (int i = 0; i < 2; i++) {
        int linear_idx = (tid + i * 256) * 8;
        int b_row = linear_idx / BN;
        int b_col = linear_idx % BN;
        
        if (b_row < K && (bx * BN + b_col) < N) {
            cp_async_cg_16(&s_B[0][b_row][b_col], &B_block[b_row * N + b_col]);
        } else {
            float4 zeros = {0.0f, 0.0f, 0.0f, 0.0f};
            *reinterpret_cast<float4*>(&s_B[0][b_row][b_col]) = zeros;
        }
    }
    cp_async_commit_group();

    for (int k = 0; k < num_k_tiles; k++) {
        int cur = k % 2;
        int nxt = (k + 1) % 2;

        if (k + 1 < num_k_tiles) {
            int next_k = (k + 1) * BK;
            for (int i = 0; i < 2; i++) {
                int linear_idx = (tid + i * 256) * 8;
                int a_row = linear_idx / BK;
                int a_col = linear_idx % BK;
                
                if ((by * BM + a_row) < M && (next_k + a_col) < K) {
                    cp_async_cg_16(&s_A[nxt][a_row][a_col], &A_block[a_row * K + next_k + a_col]);
                } else {
                    float4 zeros = {0.0f, 0.0f, 0.0f, 0.0f};
                    *reinterpret_cast<float4*>(&s_A[nxt][a_row][a_col]) = zeros;
                }
            }

            for (int i = 0; i < 2; i++) {
                int linear_idx = (tid + i * 256) * 8;
                int b_row = linear_idx / BN;
                int b_col = linear_idx % BN;
                
                if ((next_k + b_row) < K && (bx * BN + b_col) < N) {
                    cp_async_cg_16(&s_B[nxt][b_row][b_col], &B_block[(next_k + b_row) * N + b_col]);
                } else {
                    float4 zeros = {0.0f, 0.0f, 0.0f, 0.0f};
                    *reinterpret_cast<float4*>(&s_B[nxt][b_row][b_col]) = zeros;
                }
            }
            cp_async_commit_group();
        }

        cp_async_wait_group<1>();
        __syncthreads();

        for (int step = 0; step < BK; step += 16) {
            for (int i = 0; i < 2; i++) {
                int smem_m = warp_m * WM + i * 16;
                wmma::load_matrix_sync(a_frag[i], &s_A[cur][smem_m][step], BK + 8);
            }

            for (int j = 0; j < 4; j++) {
                int smem_n = warp_n * WN + j * 16;
                wmma::load_matrix_sync(b_frag[j], &s_B[cur][step][smem_n], BN + 8);
            }

            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 4; j++) {
                    wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
                }
            }
        }
        __syncthreads();
    }

    cp_async_wait_group<0>();
    __syncthreads();

    // Store c_frag to global memory
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
            int g_row = by * BM + warp_m * WM + i * 16;
            int g_col = bx * BN + warp_n * WN + j * 16;
            
            wmma::store_matrix_sync(&s_C_tile[warpId][0][0], c_frag[i][j], 16, wmma::mem_row_major);
            
            for (int t = 0; t < 8; t++) {
                int elem_idx = laneId + t * 32;
                int tile_row = elem_idx / 16;
                int tile_col = elem_idx % 16;
                
                if (g_row + tile_row < M && g_col + tile_col < N) {
                    C[(g_row + tile_row) * N + (g_col + tile_col)] = __float2half(s_C_tile[warpId][tile_row][tile_col]);
                }
            }
        }
    }
}

void launch_hgemm_v5_wmma_swizzle(const half* A, const half* B, half* C, int M, int N, int K) {
    dim3 block(256);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    hgemm_v5_wmma_swizzle_kernel<<<grid, block>>>(A, B, C, M, N, K);
    CHECK_CUDA(cudaGetLastError());
}
