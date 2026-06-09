import torch
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hgemm_tensorcore.functional import hgemm

def measure_cublas(A, B):
    # Warm up
    for _ in range(10):
        torch.mm(A, B)
        
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        torch.mm(A, B)
    end.record()
    torch.cuda.synchronize()
    
    ms = start.elapsed_time(end) / 100
    return ms

def bench(backend="v1"):
    sizes = [512, 1024, 2048, 4096, 8192]
    # Reduce max size for naive baseline because it will be extremely slow
    if backend == "v1":
        sizes = [512, 1024, 2048]

    print(f"Benchmarking {backend} backend...")
    print("N      | Our TFLOPS | cuBLAS TFLOPS | Efficiency")
    print("-------|------------|---------------|----------")

    for N in sizes:
        M = K = N
        A = torch.randn(M, K, dtype=torch.float16, device='cuda')
        B = torch.randn(K, N, dtype=torch.float16, device='cuda')

        # Warm up
        for _ in range(2):
            hgemm(A, B, backend=backend)

        # Benchmark ours
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        iters = 10 if backend == "v1" else 100
        for _ in range(iters):
            hgemm(A, B, backend=backend)
        end.record()
        torch.cuda.synchronize()

        ms_ours = start.elapsed_time(end) / iters
        tflops_ours = (2 * M * N * K) / (ms_ours * 1e6) # ms to ns, /1e9 => / 1e6

        ms_cublas = measure_cublas(A, B)
        tflops_cublas = (2 * M * N * K) / (ms_cublas * 1e6)

        efficiency = (tflops_ours / tflops_cublas) * 100

        print(f"{N:<6} | {tflops_ours:<10.2f} | {tflops_cublas:<13.2f} | {efficiency:.1f}%")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="v1")
    args = parser.parse_args()
    
    bench(args.backend)
