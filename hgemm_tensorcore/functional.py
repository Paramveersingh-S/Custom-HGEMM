import torch
import hgemm_tensorcore_cpp

# Dictionary mapping string backends to integer versions
BACKEND_TO_VERSION = {
    "naive": 1,
    "v1": 1,
    "smem": 2,
    "v2": 2,
    "wmma": 3,
    "v3": 3,
    "wmma_pipeline": 4,
    "v4": 4,
    "wmma_swizzle": 5,
    "v5": 5,
    "cute": 6,
    "v6": 6,
    "cute_swizzle": 7,
    "v7": 7,
}

def hgemm(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    backend: str = "v1",
) -> torch.Tensor:
    """
    Computes C = alpha * A @ B + beta * C using custom kernels.
    """
    assert A.dtype == torch.float16, "A must be float16"
    assert B.dtype == torch.float16, "B must be float16"
    if C is not None:
        assert C.dtype == torch.float16, "C must be float16"

    M, K = A.shape
    K_, N = B.shape
    assert K == K_, "Inner dimensions must match"

    # Pad matrices to multiples of block sizes (128 for M, N and 32 for K) 
    # to prevent out-of-bounds page faults in unpredicated CuTe copies
    pad_M = (128 - (M % 128)) % 128
    pad_N = (128 - (N % 128)) % 128
    pad_K = (32 - (K % 32)) % 32

    A_padded = torch.nn.functional.pad(A, (0, pad_K, 0, pad_M)) if (pad_K > 0 or pad_M > 0) else A
    B_padded = torch.nn.functional.pad(B, (0, pad_N, 0, pad_K)) if (pad_N > 0 or pad_K > 0) else B

    version = BACKEND_TO_VERSION.get(backend, 1)
    
    out_padded = hgemm_tensorcore_cpp.hgemm(A_padded, B_padded, version)
    out = out_padded[:M, :N].contiguous()
    
    if alpha != 1.0:
        out = out * alpha
    
    if C is not None and beta != 0.0:
        out = out + beta * C
        
    return out
