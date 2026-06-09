import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Optional: Ensure you're compiling for the correct architecture
# e.g., 'sm_75' for T4, 'sm_80' for A100, 'sm_89' for RTX 4090
# We can set a fallback list of common architectures.
os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5;8.0;8.9"

setup(
    name="hgemm_tensorcore",
    version="0.1",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="hgemm_tensorcore_cpp",
            sources=[
                "src/hgemm_dispatch.cu",
                "src/hgemm_v1_naive.cu",
                "src/hgemm_v2_smem.cu",
                "src/hgemm_v3_wmma.cu",
            ],
            include_dirs=["include"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
