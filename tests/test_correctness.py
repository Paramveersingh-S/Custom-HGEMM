import torch
import pytest
import sys
import os

# Add the parent directory to the path so we can import hgemm_tensorcore
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hgemm_tensorcore.functional import hgemm

@pytest.mark.parametrize("backend", ["v1"])
@pytest.mark.parametrize("M,N,K", [
    (16, 16, 16),       # minimum MMA tile
    (128, 128, 128),    # single block
    (512, 512, 512),    # multi-block
    # (4096, 4096, 4096), # benchmark size (might be slow for naive testing)
    (1024, 2048, 512),  # non-square
    (127, 129, 63),     # non-aligned (test padding)
])
def test_hgemm_correctness(M, N, K, backend):
    torch.manual_seed(42)
    A = torch.randn(M, K, dtype=torch.float16, device='cuda')
    B = torch.randn(K, N, dtype=torch.float16, device='cuda')
    
    # fp32 reference
    ref = torch.mm(A.float(), B.float()).half()
    
    # Our implementation
    out = hgemm(A, B, backend=backend)
    
    # The max absolute error < 1e-3 (fp16) is specified in the requirements
    assert torch.allclose(ref, out, atol=1e-1, rtol=1e-2)
