#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <time.h>
#include <math.h>
#include "../../general.h"

__global__ void convolve_volume(vec3 a_dim, float* a, size_t kernelSize, size_t out_depth, float* kernel, float* out, size_t padding, size_t stride) {
	
	int out_r = blockIdx.x;
	int out_c = blockIdx.y;
	int r_start = out_r * stride - padding;
	int c_start = out_c * stride - padding;
	int out_ch = blockIdx.z * blockDim.x + threadIdx.x;
	//int out_depth = gridDim.z;
	if (out_ch >= out_depth) return;
	float sum = 0;
	int k_offset = out_ch * kernelSize * kernelSize * a_dim.z;
	int a_size = a_dim.x * a_dim.y * a_dim.z;

	for (int i = 0, int r = r_start;i < kernelSize;i++, r++) {
		for (int j = 0, int c = c_start;j < kernelSize;j++, c++) {
			for (int ch = 0;ch < a_dim.z;ch++) {
				int k_id = k_offset + (i * kernelSize * a_dim.z) + (j * a_dim.z) + ch;
				int a_id = (r * a_dim.y * a_dim.z) + (c * a_dim.z) + ch;
				if (a_id >= 0 && a_id < a_size)
					sum += a[a_id] * kernel[k_id];
			}
		}
	}
	int out_id = (out_r * gridDim.y * out_depth) + (out_c * out_depth) + out_ch;
	out[out_id] = sum;
}

__global__ void max_pool(vec3 a_dim, float* a, size_t poolSize, float* out, size_t padding, size_t stride) {
	int out_r = blockIdx.x;
	int out_c = blockIdx.y;
	int r_start = out_r * stride - padding;
	int c_start = out_c * stride - padding;
	int chan = blockIdx.z * blockDim.x + threadIdx.x;
	//int out_depth = gridDim.z;
	if (chan >= a_dim.z) return;
	int a_size = a_dim.x * a_dim.y * a_dim.z;

	float max = -FLT_MAX;
	for (int i = 0, int r = r_start;i < poolSize;i++, r++) {
		for (int j = 0, int c = c_start;j < poolSize;j++, c++) {
			int a_id = (r * a_dim.y * a_dim.z) + (c * a_dim.z) + chan;
			if (a_id >= 0 && a_id < a_size) {
				float term = a[a_id];
				if (term > max) max = term;
			}
		}
	}
	int out_id = (out_r * gridDim.y * a_dim.z) + (out_c * a_dim.z) + chan;
	out[out_id] = max;
}