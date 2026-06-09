#!/bin/bash
# Script to run on Google Colab to build the extension and run benchmarks

echo "Installing pytest..."
pip install pytest

if [ ! -d "cutlass" ]; then
    echo "Cloning CUTLASS for CuTe headers..."
    git clone https://github.com/NVIDIA/cutlass.git
fi

echo "Building PyTorch extension (verbose mode)..."
pip install -v -e .

echo "Running correctness tests..."
pytest tests/test_correctness.py -v

echo "Running benchmark..."
python benchmarks/bench_vs_cublas.py --backend v3
