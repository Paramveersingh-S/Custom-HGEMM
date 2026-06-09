#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>

#define CHECK_CUDA(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " code=" << err << " \""                        \
                      << cudaGetErrorString(err) << "\"" << std::endl;   \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)
