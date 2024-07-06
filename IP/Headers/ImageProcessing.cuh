#include "../../CUDA/Headers/general.cuh"
#include <cstring>
#include <time.h>
#include <math.h>

dim3 getDims(dim3 a_dim, size_t k_size, size_t padding, size_t stride, size_t out_depth);

float* convolve(dim3 a_dim, float* a, dim3 k_dim, int out_depth, float* kernel, int padding, int stride);

float* convolve_volume(dim3 a_dim, float* a, dim3 k_dim, size_t out_depth, float* kernel, size_t padding, size_t stride);

float* max_pool(float* a, dim3 a_dim, size_t pool_size, size_t padding, size_t stride);