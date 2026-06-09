#include <torch/extension.h>
#include <cuda_fp16.h>
#include <string>

// Forward declarations of launch functions
void launch_hgemm_v1_naive(const half* A, const half* B, half* C, int M, int N, int K);
void launch_hgemm_v2_smem(const half* A, const half* B, half* C, int M, int N, int K);
void launch_hgemm_v3_wmma(const half* A, const half* B, half* C, int M, int N, int K);

// The main dispatcher
torch::Tensor hgemm_forward(torch::Tensor A, torch::Tensor B, int version) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.scalar_type() == torch::kFloat16, "A must be float16");
    TORCH_CHECK(B.scalar_type() == torch::kFloat16, "B must be float16");
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "Inner dimensions must match");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(A.device());
    torch::Tensor C = torch::empty({M, N}, options);

    const half* ptr_A = reinterpret_cast<const half*>(A.data_ptr<at::Half>());
    const half* ptr_B = reinterpret_cast<const half*>(B.data_ptr<at::Half>());
    half* ptr_C = reinterpret_cast<half*>(C.data_ptr<at::Half>());

    switch(version) {
        case 1:
            launch_hgemm_v1_naive(ptr_A, ptr_B, ptr_C, M, N, K);
            break;
        case 2:
            launch_hgemm_v2_smem(ptr_A, ptr_B, ptr_C, M, N, K);
            break;
        case 3:
            launch_hgemm_v3_wmma(ptr_A, ptr_B, ptr_C, M, N, K);
            break;
        default:
            TORCH_CHECK(false, "Unsupported version");
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hgemm", &hgemm_forward, "HGEMM forward");
}
