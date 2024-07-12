#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <time.h>
#include <math.h>
#include "../../general.h"
#include "../../MLP/Headers/MLP.h"

// Declare the functions
__global__ void gpu_convolve_volume(vec3 a_dim, float* a, size_t kernelSize, size_t out_depth, float* kernel, float* out, size_t padding, size_t stride);

__global__ void size_convolve_volume(vec3 a_dim, float* a, size_t kernelSize, vec3 out_dim, float* kernel, float* out, size_t padding, size_t stride);

__global__ void gpu_max_pool(vec3 a_dim, float* a, size_t poolSize, float* out, size_t padding, size_t stride);

__global__ void size_max_pool(vec3 a_dim, float* a, size_t poolSize, float* out, size_t padding, size_t stride, vec3 out_dim);

__global__ void gpu_activate(float* a, activation_function A_F, int size);

__global__ void gpu_backVolve(float* a, vec3 a_dim, float* d_terms, int kernelSize, float* d_kernel, vec3 out_dim, float* d_for_terms, size_t padding, size_t stride, activation_function A_F);

__global__ void gpu_setZero(float* a, int size);

__global__ void gpu_backPool(vec3 a_dim, float* a, float* terms, vec3 out_dim, float* for_terms, size_t pool_size, size_t padding, size_t stride);

__global__ void gpu_gradVolve(vec3 a_dim, float* in_layer, vec3 k_dim, float* gradient, vec3 out_dim, float* out_terms, size_t padding, size_t stride);
