import matplotlib.pyplot as plt
import numpy as np

# Data from the user's benchmark run
sizes_v1 = [512, 1024, 2048]
tflops_v1 = [245.36, 250.22, 430.55]
eff_v1 = [3.1, 1.6, 1.9]

sizes = [512, 1024, 2048, 4096, 8192]
tflops_v2 = [1039.35, 1736.18, 3150.24, 3141.81, 3011.91]
eff_v2 = [11.0, 9.0, 12.5, 15.0, 15.8]

tflops_v3 = [1686.46, 2017.52, 3151.86, 3410.60, 3310.58]
eff_v3 = [14.6, 8.5, 13.8, 21.0, 16.7]

tflops_v4 = [3367.80, 4782.04, 6505.57, 8738.43, 8774.83]
eff_v4 = [30.4, 20.8, 27.4, 47.6, 44.3]

tflops_v5 = [5286.99, 10339.04, 13192.88, 12835.48, 13312.27]
eff_v5 = [45.9, 43.5, 70.2, 69.6, 68.6]

tflops_v6 = [3209.39, 4433.28, 6714.48, 8131.21, 8024.88]
eff_v6 = [30.5, 20.6, 31.4, 45.6, 41.6]

tflops_v7 = [3099.62, 4847.03, 7430.78, 10665.78, 10758.46]
eff_v7 = [37.6, 30.3, 37.1, 58.9, 55.4]

# cuBLAS (from v5 as baseline)
cublas = [11508.90, 23766.45, 18797.16, 18441.26, 19394.10]

# --- Plot 1: Absolute TFLOPS ---
plt.figure(figsize=(10, 6), dpi=150)
plt.plot(sizes_v1, tflops_v1, marker='o', label='v1 (Naive)', linestyle='--')
plt.plot(sizes, tflops_v2, marker='o', label='v2 (SMEM Tiling)')
plt.plot(sizes, tflops_v3, marker='o', label='v3 (WMMA)')
plt.plot(sizes, tflops_v4, marker='o', label='v4 (Async Pipeline)')
plt.plot(sizes, tflops_v5, marker='o', label='v5 (SMEM Padded)', linewidth=2.5)
plt.plot(sizes, tflops_v6, marker='o', label='v6 (CuTe)')
plt.plot(sizes, tflops_v7, marker='o', label='v7 (CuTe + Swizzle)')
plt.plot(sizes, cublas, marker='*', label='cuBLAS (Target)', color='black', linewidth=2, linestyle=':')

plt.title('Performance Comparison (Tesla T4 SM75)', fontsize=14, fontweight='bold')
plt.xlabel('Matrix Size (M=N=K)', fontsize=12)
plt.ylabel('TFLOPS', fontsize=12)
plt.xscale('log', base=2)
plt.xticks(sizes, [str(s) for s in sizes])
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('docs/tflops_comparison.png')

# --- Plot 2: Efficiency vs cuBLAS ---
plt.figure(figsize=(10, 6), dpi=150)
plt.plot(sizes_v1, eff_v1, marker='o', label='v1 (Naive)', linestyle='--')
plt.plot(sizes, eff_v2, marker='o', label='v2 (SMEM Tiling)')
plt.plot(sizes, eff_v3, marker='o', label='v3 (WMMA)')
plt.plot(sizes, eff_v4, marker='o', label='v4 (Async Pipeline)')
plt.plot(sizes, eff_v5, marker='o', label='v5 (SMEM Padded)', linewidth=2.5)
plt.plot(sizes, eff_v6, marker='o', label='v6 (CuTe)')
plt.plot(sizes, eff_v7, marker='o', label='v7 (CuTe + Swizzle)')

plt.axhline(y=100, color='black', linestyle=':', label='cuBLAS (100%)')

plt.title('cuBLAS Efficiency % (Tesla T4 SM75)', fontsize=14, fontweight='bold')
plt.xlabel('Matrix Size (M=N=K)', fontsize=12)
plt.ylabel('Efficiency (%)', fontsize=12)
plt.xscale('log', base=2)
plt.xticks(sizes, [str(s) for s in sizes])
plt.ylim(0, 105)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('docs/efficiency_comparison.png')
