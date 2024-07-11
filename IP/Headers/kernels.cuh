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
				if (a_id >= 0 && a_id < a_size) {
					sum += a[a_id] * kernel[k_id];
				}
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

__global__ void activate(float* a, activation_function A_F, int size) {
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	float val = a[gid];
	switch (A_F) {
	case SIGMOID:
		val = 1 / (1 + exp(-val)); //sigmoid
		break;
	case RELU:
		if (val <= 0) val = 0; //ReLu
		break;
	case LEAKY_RELU:
		if (val <= 0) val *= .1; //Leaky ReLu
		break;
	}
}

//Maybe try thinking of it as true reverse convolution where for_terms is the kernel and terms is the inner product of the sub-section and kernel
__global__ void backVolve(float* a, vec3 a_dim, float* d_terms, int kernelSize, float* d_kernel, vec3 out_dim, float* d_for_terms, size_t padding, size_t stride, activation_function A_F) {
	size_t a_id = blockIdx.x * blockDim.x + threadIdx.x;
	size_t a_size = a_dim.x * a_dim.y * a_dim.z;

	if (a_id >= a_size) return;

	int channels = a_dim.z;
	int height = out_dim.x;
	int width = out_dim.y;

	int z = a_id % channels;
	int xy = a_id / channels;
	int y = xy % a_dim.y;
	int x = xy / a_dim.y;

	float sum = 0;

	for (int i = 0; i < kernelSize; ++i) {
		for (int j = 0; j < kernelSize; ++j) {
			int r = x - padding + i;
			int c = y - padding + j;
			if (r >= 0 && r < height && c >= 0 && c < width) {
				for (int out_chan = 0; out_chan < out_dim.z; ++out_chan) {
					int out_r = (r + padding) / stride;
					int out_c = (c + padding) / stride;
					if (out_r < height && out_c < width) {
						sum += d_kernel[(out_chan * kernelSize * kernelSize * channels) + (i * kernelSize * channels) + (j * channels) + z] * d_for_terms[out_r * width * out_dim.z + out_c * out_dim.z + out_chan];
					}
				}
			}
		}
	}
	float output = a[a_id];
	if (A_F == SIGMOID) sum = output * (1 - output) * sum;
	else if (A_F == RELU) sum = (output > 0) ? sum : 0;
	else if (A_F == LEAKY_RELU) sum = (output > 0) ? sum : (.1 * sum);
	d_terms[a_id] = sum;
}

__global__ void setZero(float* a, int size) {
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	a[gid] = 0;
}

//NEED TO SET TERMS TO 0 BEFORE BACKPOOLING

__global__ void backPool(vec3 a_dim, float* a, float* terms, vec3 out_dim, float* for_terms, size_t pool_size, size_t padding, size_t stride) {
	size_t o_id = blockIdx.x * blockDim.x + threadIdx.x;
	size_t o_size = out_dim.x * out_dim.y * out_dim.z;

	if (o_id >= o_size) return;

	int chan = o_id % out_dim.z;
	int rc = o_id / out_dim.z;
	int out_c = rc % out_dim.y;
	int out_r = rc / out_dim.y;

	int r_start = out_r * stride - padding;
	int c_start = out_c * stride - padding;
	//int out_depth = gridDim.z;
	int a_size = a_dim.x * a_dim.y * a_dim.z;

	float max = -INFINITY;
	int id;
	for (int i = 0, int r = r_start;i < pool_size;i++, r++) {
		for (int j = 0, int c = c_start;j < pool_size;j++, c++) {
			int a_id = (r * a_dim.y * a_dim.z) + (c * a_dim.z) + chan;
			if (a_id >= 0 && a_id < a_size) {
				float term = a[a_id];
				if (a[a_id] > max) id = a_id;
			}
		}
	}
	int out_id = (out_r * gridDim.y * a_dim.z) + (out_c * a_dim.z) + chan;
	terms[id] = for_terms[out_id];
}

__global__ void gradVolve(vec3 a_dim, float* in_layer, vec3 k_dim, float* gradient, vec3 out_dim, float* out_terms, size_t padding, size_t stride) {

	int g_id = blockIdx.x * gridDim.y * blockDim.x + blockIdx.y * blockDim.x + threadIdx.x;
	if (g_id >= out_dim.z * k_dim.x * k_dim.y * k_dim.z) return;
	int out_channels = out_dim.z;

	int in_chan = g_id % k_dim.z;
	int cij = g_id / k_dim.z;

	int j = cij % k_dim.y;
	int ci = cij / k_dim.y;

	int i = ci % k_dim.x;
	//int out_chan = ci / k_dim.x;
	int out_chan = blockIdx.x;

	int height = out_dim.x;
	int width = out_dim.y;

	int kernelSize = k_dim.x;
	int channels = a_dim.z;

	size_t a_size = a_dim.x * a_dim.y * a_dim.z;
	int end = (int)((k_dim.x / 2.0) + .5);
	int start = end - k_dim.x;
	float sum;
	for (int r = -padding, out_r = 0; out_r < height; r += stride, ++out_r) {
		for (int c = -padding, out_c = 0; out_c < width; c += stride, ++out_c) {
			for (int chan = 0;chan < channels;chan++) {
				int a_id = ((r + i) * a_dim.y * a_dim.z) + ((c + j) * a_dim.z) + chan;
				if (a_id >= 0 && a_id < a_size) {
					gradient[g_id] += in_layer[a_id] * out_terms[out_r * width * out_dim.z + out_c * out_dim.z + out_chan];
				}
			}
		}
	}
}