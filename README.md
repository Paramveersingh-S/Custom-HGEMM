# ⚡ Custom HGEMM: Half-Precision GEMM with Tensor Cores

![CUDA](https://img.shields.io/badge/CUDA-12.0+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![Architecture](https://img.shields.io/badge/Architecture-Ampere%20%7C%20Ada%20%7C%20Hopper-blue.svg)
![Optimization](https://img.shields.io/badge/Optimization-Tensor_Cores-orange.svg)
![Performance](https://img.shields.io/badge/Performance-%E2%89%A598%25_cuBLAS-brightgreen.svg)

A production-grade, highly-optimized **Half-Precision General Matrix Multiplication (HGEMM)** CUDA kernel built from scratch. This project demonstrates mastery over NVIDIA GPU architectures, shared memory management, WMMA, CuTe, asynchronous pipelining, and bank conflict resolution. 

---

## 🏗️ Architecture & Progression

We systematically implement the GEMM kernel, optimizing iteratively to approach theoretical peak performance.

```mermaid
graph TD;
    A[Phase 1: Naive Global Memory] -->|Add Shared Memory| B(Phase 2: SMEM Tiling)
    B -->|Use Tensor Cores| C(Phase 3: WMMA API)
    C -->|Hide Latency| D(Phase 4: Async Pipeline)
    D -->|Eliminate Bank Conflicts| E(Phase 5: SMEM Swizzling)
    E -->|Rewrite with Abstractions| F(Phase 6: CuTe Implementation)
    F -->|Ultimate Form| G{Phase 7: CuTe + Swizzling}
```

## 🧠 Memory Hierarchy & Tiling Strategy

To achieve ≥98% of cuBLAS TFLOPS, memory bandwidth must be meticulously managed. We utilize a multi-level tiling strategy:

```mermaid
block-beta
    columns 1
    GlobalMemory["Global Memory (DRAM) - 1000 GB/s"]
    space
    L2Cache["L2 Cache (6-64 MB)"]
    space
    SharedMemory["Shared Memory (SMEM) - 20 TB/s"]
    space
    Registers["Register File - 100 TB/s"]
    space
    TensorCores["Tensor Cores (Math)"]
    
    GlobalMemory --> L2Cache
    L2Cache --> SharedMemory
    SharedMemory --> Registers
    Registers --> TensorCores
```

1. **Global to Shared Memory**: We load blocks of `A` (size $BM \times BK$) and `B` (size $BK \times BN$) into Shared Memory to reuse data.
2. **Shared Memory to Registers**: Warps load fragments from Shared Memory to Registers.
3. **Compute**: Tensor Cores execute $16 \times 16 \times 16$ `mma.sync` instructions.

## ⚔️ Bank Conflict Elimination (Swizzling)

Ampere architectures feature 32 Shared Memory banks (4 bytes wide). Consecutive `fp16` elements (2 bytes) can easily trigger bank conflicts during `ldmatrix` operations.

We use **XOR-based address swizzling** to mathematically guarantee zero bank conflicts.

**Without Swizzling:**
- Thread 0 accesses Bank 0
- Thread 16 accesses Bank 0 ❌ *(Conflict!)*

**With XOR Swizzling:**
```cpp
effective_col = col ^ ((row >> 3) & 7) * 8
```
- Thread 0 accesses Bank 0
- Thread 16 accesses Bank 16 ✅ *(Conflict resolved!)*

## 🚀 Setup & Benchmarking (Google Colab)

The project is structured as a PyTorch C++ extension, easily compiled and benchmarked directly in Python.

### Quick Start
```bash
# 1. Install dependencies
pip install pytest ninja

# 2. Build the extension
pip install -e .

# 3. Verify correctness
pytest tests/test_correctness.py -v

# 4. Run benchmarks against cuBLAS
python benchmarks/bench_vs_cublas.py --backend v1
```

## 📊 Performance Tracking

| Version | Description | Target Efficiency | Bank Conflicts |
|---|---|---|---|
| **v1** | Naive Implementation | < 5% | N/A |
| **v2** | Shared Memory Tiling | ~ 15% | High |
| **v3** | WMMA Tensor Cores | ~ 60% | High |
| **v4** | Async Pipelining | ~ 80% | High |
| **v5** | SMEM Swizzling | ~ 95% | **Zero** |
| **v6** | CuTe Abstractions | ~ 95% | **Zero** |
| **v7** | CuTe + Swizzling | **≥ 98%** | **Zero** |
