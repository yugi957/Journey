#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <time.h>
#include <math.h>


__global__ void kernel9D(int* input, int size) {
	int i = blockIdx.z * gridDim.y * gridDim.x * blockDim.z * blockDim.y * blockDim.x + blockIdx.y * gridDim.x * blockDim.z * blockDim.y * blockDim.x + blockIdx.x * blockDim.z * blockDim.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;

	if (i < size) printf("arr[%d] = %d\n", i, input[i]);
}

__global__ void sum_array(int* a, int* b, int* c, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < size) c[i] = a[i] + b[i];
}

__global__ void sum_arrays(int* a, int* b, int* c, int* d, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < size) d[i] = a[i] + b[i] + c[i];
}

__global__ void warpDetails() {
	int i = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;

	int bId = blockIdx.y * gridDim.x + blockIdx.x;

	
	int WPB = blockDim.x / 32;
	if(blockDim.x % 32 != 0) WPB++;
	int warp_id = (threadIdx.x / 32) + bId * WPB;


	printf("thread[%d][%d][%d]::: thread == %d : block == %d : warp == %d\n", blockIdx.x, blockIdx.y, threadIdx.x, i, bId, warp_id);
}

__global__ void no_divergence() {
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	float a, b;
	a = b = 0;

	int warp_id = gid / 32;

	if (warp_id % 2 == 0) {
		a = 100.0;
		b = 50.0;
	}
	else {
		a = 200.0;
		b = 75.0;
	}
}

__global__ void divergent() {
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	float a, b;
	a = b = 0;

	if (gid % 2 == 0) {
		a = 100.0;
		b = 50.0;
	}
	else {
		a = 200.0;
		b = 75.0;
	}
}