#include "hgemm_common.cuh"

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cute/tensor.hpp>
#include <cute/algorithm/gemm.hpp>

using namespace cute;

// Tile shapes
using bM = Int<128>;
using bN = Int<128>;
using bK = Int<32>;

// Thread layout for MMA: 4x2x1 warps (256 threads), covering the full bM x bN x bK tile
using MMA_Atom_Arch = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;
using TiledMMA_Arch = TiledMMA<MMA_Atom_Arch, Layout<Shape<_4,_2,_1>>, Tile<bM,bN,bK>>;

// Shared memory layouts (No swizzle for v6)
using SmemLayoutA = decltype(make_layout(make_shape(bM{}, bK{}), LayoutRight{}));
using SmemLayoutB = decltype(make_layout(make_shape(bN{}, bK{}), LayoutRight{}));

__global__ void hgemm_v6_cute_kernel(const half* A, const half* B, half* C, int M, int N, int K) {
    // 1. Tensors in Global Memory
    // A is M x K row-major
    Tensor gA = make_tensor(make_gmem_ptr(A), make_shape(M, K), make_stride(K, Int<1>{}));
    // B is K x N row-major -> B^T is N x K col-major
    Tensor gB = make_tensor(make_gmem_ptr(B), make_shape(N, K), make_stride(Int<1>{}, N));
    // C is M x N row-major
    Tensor gC = make_tensor(make_gmem_ptr(C), make_shape(M, N), make_stride(N, Int<1>{}));

    // Block coordinates
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Slice global tensors for the current block
    Tensor gA_block = local_tile(gA, make_tile(bM{}, bK{}), make_coord(by, _)); // (bM, bK, k_tiles)
    Tensor gB_block = local_tile(gB, make_tile(bN{}, bK{}), make_coord(bx, _)); // (bN, bK, k_tiles)
    Tensor gC_block = local_tile(gC, make_tile(bM{}, bN{}), make_coord(by, bx)); // (bM, bN)

    // 2. Shared Memory Tensors
    __shared__ half sA_data[cosize_v<SmemLayoutA>];
    __shared__ half sB_data[cosize_v<SmemLayoutB>];

    Tensor sA = make_tensor(make_smem_ptr(sA_data), SmemLayoutA{}); // (bM, bK)
    Tensor sB = make_tensor(make_smem_ptr(sB_data), SmemLayoutB{}); // (bN, bK)

    // 3. Tiled Copy for Global to Shared (cp.async)
    using CopyAtom = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, half>;
    auto tiled_copy_A = make_tiled_copy(CopyAtom{},
                                        Layout<Shape<_32, _8>, Stride<_8, _1>>{},
                                        Layout<Shape<_1, _8>>{});
    auto tiled_copy_B = make_tiled_copy(CopyAtom{},
                                        Layout<Shape<_32, _8>, Stride<_8, _1>>{},
                                        Layout<Shape<_1, _8>>{});

    auto thr_copy_A = tiled_copy_A.get_thread_slice(threadIdx.x);
    auto thr_copy_B = tiled_copy_B.get_thread_slice(threadIdx.x);

    Tensor tAgA = thr_copy_A.partition_S(gA_block); // (CPY, CPY_M, CPY_K, k_tiles)
    Tensor tAsA = thr_copy_A.partition_D(sA);       // (CPY, CPY_M, CPY_K)
    Tensor tBgB = thr_copy_B.partition_S(gB_block); // (CPY, CPY_N, CPY_K, k_tiles)
    Tensor tBsB = thr_copy_B.partition_D(sB);       // (CPY, CPY_N, CPY_K)

    // 4. Thread-level MMA and Registers
    TiledMMA_Arch tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
    
    Tensor tCsA = thr_mma.partition_A(sA); // (MMA, MMA_M, MMA_K)
    Tensor tCsB = thr_mma.partition_B(sB); // (MMA, MMA_N, MMA_K)
    Tensor tCgC = thr_mma.partition_C(gC_block); // (MMA, MMA_M, MMA_N)

    // Accumulator
    Tensor tCrC = thr_mma.make_fragment_C(tCgC); // (MMA, MMA_M, MMA_N) register
    clear(tCrC);

    // K-Loop
    int num_k_tiles = size<2>(gA_block);

    for (int k = 0; k < num_k_tiles; ++k) {
        // Copy to SMEM
        cute::copy(tiled_copy_A, tAgA(_, _, _, k), tAsA);
        cute::copy(tiled_copy_B, tBgB(_, _, _, k), tBsB);
        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();

        // GEMM on SMEM
        Tensor tCrA = thr_mma.make_fragment_A(tCsA);
        Tensor tCrB = thr_mma.make_fragment_B(tCsB);

        cute::copy(tCsA, tCrA);
        cute::copy(tCsB, tCrB);
        cute::gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC);
        
        __syncthreads();
    }

    // Convert accumulator and store to global memory
    Tensor tCgC_half = make_tensor(tCgC.data(), tCgC.shape(), tCgC.stride());
    for (int i = 0; i < size(tCrC); ++i) {
        tCgC_half(i) = __float2half(tCrC(i));
    }
}
#endif // __CUDA_ARCH__ >= 800

void launch_hgemm_v6_cute(const half* A, const half* B, half* C, int M, int N, int K) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    dim3 block(256);
    dim3 grid((N + 128 - 1) / 128, (M + 128 - 1) / 128);

    hgemm_v6_cute_kernel<<<grid, block>>>(A, B, C, M, N, K);
    CHECK_CUDA(cudaGetLastError());
#else
    std::cerr << "CuTe requires SM80+" << std::endl;
#endif
}
