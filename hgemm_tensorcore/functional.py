import torch
import hgemm_tensorcore_cpp

# Dictionary mapping string backends to integer versions
BACKEND_TO_VERSION = {
    "naive": 1,
    "v1": 1,
    # "smem": 2,
    # "v2": 2,
    # "wmma": 3,
    # "v3": 3,
    # "wmma_pipeline": 4,
    # "v4": 4,
    # "wmma_swizzle": 5,
    # "v5": 5,
    # "cute": 6,
    # "v6": 6,
    # "cute_swizzle": 7,
    # "v7": 7,
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

    version = BACKEND_TO_VERSION.get(backend, 1)
    
    out = hgemm_tensorcore_cpp.hgemm(A, B, version)
    
    if alpha != 1.0:
        out = out * alpha
    
    if C is not None and beta != 0.0:
        out = out + beta * C
        
    return out
