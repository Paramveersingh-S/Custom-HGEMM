#!/bin/bash
# Script to run on Google Colab to build the extension and run benchmarks

echo "Installing pytest..."
pip install pytest

echo "Building PyTorch extension..."
pip install -e .

echo "Running correctness tests..."
pytest tests/test_correctness.py -v

echo "Running benchmark..."
python benchmarks/bench_vs_cublas.py --backend v3
