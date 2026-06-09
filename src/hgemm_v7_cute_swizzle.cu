#include "hgemm_common.cuh"

#include <cute/tensor.hpp>
#include <cute/algorithm/gemm.hpp>

using namespace cute;

__global__ void hgemm_v7_cute_swizzle_kernel(const half* A, const half* B, half* C, int M, int N, int K) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    // Tile shapes
    using bM = Int<128>;
    using bN = Int<128>;
    using bK = Int<32>;

    // Thread layout for MMA
    using MMA_Atom_Arch = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;
    using TiledMMA_Arch = TiledMMA<MMA_Atom_Arch, Layout<Shape<_4,_2,_1>>, Tile<bM,bN,bK>>;

    // Shared memory swizzle atoms (Swizzle<3,4,3> preserves 16-byte contiguity for cp.async!)
    using SmemLayoutAtomA = decltype(composition(Swizzle<3,4,3>{}, Layout<Shape<_8, _64>, Stride<_64, _1>>{})); // K contiguous
    using SmemLayoutAtomB = decltype(composition(Swizzle<3,4,3>{}, Layout<Shape<_64, _8>, Stride<_1, _64>>{})); // N contiguous

    using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtomA{}, make_shape(bM{}, bK{})));
    using SmemLayoutB = decltype(tile_to_shape(SmemLayoutAtomB{}, make_shape(bN{}, bK{})));

    Tensor gA = make_tensor(make_gmem_ptr(A), make_shape(M, K), make_stride(K, Int<1>{}));
    Tensor gB = make_tensor(make_gmem_ptr(B), make_shape(N, K), make_stride(Int<1>{}, N));
    Tensor gC = make_tensor(make_gmem_ptr(C), make_shape(M, N), make_stride(N, Int<1>{}));

    int bx = blockIdx.x;
    int by = blockIdx.y;

    Tensor gA_block = local_tile(gA, make_tile(bM{}, bK{}), make_coord(by, _));
    Tensor gB_block = local_tile(gB, make_tile(bN{}, bK{}), make_coord(bx, _));
    Tensor gC_block = local_tile(gC, make_tile(bM{}, bN{}), make_coord(by, bx));

    __shared__ half sA_data[cosize_v<SmemLayoutA>];
    __shared__ half sB_data[cosize_v<SmemLayoutB>];

    Tensor sA = make_tensor(make_smem_ptr(sA_data), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(sB_data), SmemLayoutB{});

    using CopyAtom = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, half>;
    auto tiled_copy_A = make_tiled_copy(CopyAtom{},
                                        Layout<Shape<_64, _4>, Stride<_4, _1>>{},
                                        Layout<Shape<_1, _8>>{});
    auto tiled_copy_B = make_tiled_copy(CopyAtom{},
                                        Layout<Shape<_16, _16>, Stride<_1, _16>>{},
                                        Layout<Shape<_8, _1>>{});

    auto thr_copy_A = tiled_copy_A.get_thread_slice(threadIdx.x);
    auto thr_copy_B = tiled_copy_B.get_thread_slice(threadIdx.x);

    Tensor tAgA = thr_copy_A.partition_S(gA_block);
    Tensor tAsA = thr_copy_A.partition_D(sA);
    Tensor tBgB = thr_copy_B.partition_S(gB_block);
    Tensor tBsB = thr_copy_B.partition_D(sB);

    TiledMMA_Arch tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
    
    Tensor tCsA = thr_mma.partition_A(sA);
    Tensor tCsB = thr_mma.partition_B(sB);
    Tensor tCgC = thr_mma.partition_C(gC_block);

    Tensor tCrC = thr_mma.make_fragment_C(tCgC);
    clear(tCrC);

    int num_k_tiles = size<2>(gA_block);

    for (int k = 0; k < num_k_tiles; ++k) {
        cute::copy(tiled_copy_A, tAgA(_, _, _, k), tAsA);
        cute::copy(tiled_copy_B, tBgB(_, _, _, k), tBsB);
        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();

        Tensor tCrA = thr_mma.make_fragment_A(tCsA);
        Tensor tCrB = thr_mma.make_fragment_B(tCsB);

        cute::copy(tCsA, tCrA);
        cute::copy(tCsB, tCrB);
        cute::gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC);
        
        __syncthreads();
    }

    Tensor tCgC_half = make_tensor(tCgC.data(), tCgC.shape(), tCgC.stride());
    for (int i = 0; i < size(tCrC); ++i) {
        tCgC_half(i) = __float2half(tCrC(i));
    }
#endif
}

void launch_hgemm_v7_cute_swizzle(const half* A, const half* B, half* C, int M, int N, int K) {
    dim3 block(256);
    dim3 grid((N + 128 - 1) / 128, (M + 128 - 1) / 128);

    hgemm_v7_cute_swizzle_kernel<<<grid, block>>>(A, B, C, M, N, K);
    CHECK_CUDA(cudaGetLastError());
}
