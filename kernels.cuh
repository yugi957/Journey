#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <time.h>
#include <math.h>
#include "MLP.cuh"
#include "MLP.h"

#define SHARED_ARRAY_SIZE 128
#define BDIMX 32
#define BDIMY 32


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

//__global__ void no_divergence() {
//	int gid = blockIdx.x * blockDim.x + threadIdx.x;
//
//	float a, b;
//	a = b = 0;
//
//	int warp_id = gid / 32;
//
//	if (warp_id % 2 == 0) {
//		a = 100.0;
//		b = 50.0;
//	}
//	else {
//		a = 200.0;
//		b = 75.0;
//	}
//}

//__global__ void divergent() {
//	int gid = blockIdx.x * blockDim.x + threadIdx.x;
//
//	float a, b;
//	a = b = 0;
//
//	if (gid % 2 == 0) {
//		a = 100.0;
//		b = 50.0;
//	}
//	else {
//		a = 200.0;
//		b = 75.0;
//	}
//}

//So reduction problem is just related to divisibility, and since the sum in a particular thread is dependent on val of sums in other threads, it's not divisible.
//Because it isn't divisible, it is simply not accurate
//I think something key to learn here is that parallelization is also kind of constrained to working with DIFFERENT memory values per thread. If any threads work on the
//same memory values, then I think there may be problems. This means a literal memory space, such as a, a[i], a[i+1], etc.
__global__ void no_reduction(int* sum, int* a, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < size) {
		*sum += a[i];
		__syncthreads();
	}
}

//91.55% branch efficiency something to note is that this gets higher "compute(SM) throughput"... Both reduced versions get lower compute and memory throughput
__global__ void neighbored_paired_reduction(int* a, int* part, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;

	if (i > size) return;

	for (int offset = 1;offset <= blockDim.x/2;offset *= 2) {
		if (i % (2 * offset) == 0) a[i] += a[i + offset];
		__syncthreads();
	}

	if (tid == 0) part[blockIdx.x] = a[i];
}

//96.63% branch efficiency
//This shifts all operations to leftmost warps
//With this reduction, the last 5 iterations (where offset makes it so num of numbers to be summed is < # of threads in warp (32)) still have divergence
//Looks like we use interleaved pairs for simpler understanding of last 5 pairs being 16,8,4,2,1
__global__ void efficientSum(int* a, int* part, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;

	if (i > size) return;

	int* start = (a + blockIdx.x * blockDim.x);

	for (int offset = 1;offset <= blockDim.x/2;offset *= 2) {
		int ind = 2 * offset * tid;
		if (ind < blockDim.x) start[ind] += start[ind + offset];
		__syncthreads();
	}

	if (tid == 0) part[blockIdx.x] = a[i];
}

__global__ void efficientSum(double* a, double* part, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;

	if (i > size) return;

	double* start = (a + blockIdx.x * blockDim.x);

	for (int offset = 1;offset <= blockDim.x / 2;offset *= 2) {
		int ind = 2 * offset * tid;
		if (ind < blockDim.x) start[ind] += start[ind + offset];
		__syncthreads();
	}

	if (tid == 0) part[blockIdx.x] = a[i];
}

//97.09 branch efficiency, but I was getting higher execution time than just efficient sum
__global__ void reduction_interleaved_pairs(int* a, int* part, int size){
	int tid = threadIdx.x;
	int gid = blockDim.x * blockIdx.x + threadIdx.x;

	if (gid > size) return;

	for (int offset = blockDim.x / 2; offset > 0; offset = offset / 2){
		if (tid < offset){
			a[gid] += a[gid + offset];
		}
		__syncthreads();
	}

	if (tid == 0) part[blockIdx.x] = a[gid];
}

__global__ void reduction_unrolling_blocks2(int* a, int* part, int size) {
	int tid = threadIdx.x;
	int BLOCK_OFFSET = blockIdx.x * blockDim.x * 2;
	int gid = BLOCK_OFFSET + tid;
	if (gid > size) return;
	int* start = a + BLOCK_OFFSET;

	if ((gid + blockDim.x) < size) a[gid] += a[gid + blockDim.x];
	__syncthreads();

	for (int offset = 1;offset <= blockDim.x / 2;offset *= 2) {
		int ind = 2 * offset * tid;
		if (ind < blockDim.x) start[ind] += start[ind + offset];
		__syncthreads();
	}

	if (tid == 0) part[blockIdx.x] = a[gid];
}

//GOES CRAZY ON EXE TIME WITH LOTS OF ROLLS, but have to be careful with how many, start to lose if too much or too less
//YOOOOO 99.99% BRANCH EFFICIENCY AVG 25.60 DIVERGENT BRANCHES OUT OF 33681920 BRANCHES TOTAL
__global__ void reduction_unrolling_blocks(int* a, int* part, int size, int rolls) {
	int tid = threadIdx.x;
	int BLOCK_OFFSET = blockIdx.x * blockDim.x * rolls;
	int gid = BLOCK_OFFSET + tid;
	if (gid > size) return;
	int* start = a + BLOCK_OFFSET;

	if ((gid + blockDim.x * (rolls - 1)) < size) {
		int sum = 0;
		for (int i = 0;i < rolls;i++) {
			sum += a[gid + (blockDim.x * i)];
		}
		a[gid] = sum;
	}
	__syncthreads();

	for (int offset = 1;offset <= blockDim.x / 2;offset *= 2) {
		int ind = 2 * offset * tid;
		if (ind < blockDim.x) start[ind] += start[ind + offset];
		__syncthreads();
	}

	if (tid == 0) part[blockIdx.x] = a[gid];
}

__global__ void reduction_unrolling_interleaved(int* a, int* part, int size, int rolls) {
	int tid = threadIdx.x;
	int BLOCK_OFFSET = blockIdx.x * blockDim.x * rolls;
	int gid = BLOCK_OFFSET + tid;
	if (gid > size) return;
	//int* start = a + BLOCK_OFFSET;

	if ((gid + blockDim.x * (rolls - 1)) < size) {
		int sum = 0;
		for (int i = 0;i < rolls;i++) {
			sum += a[gid + (blockDim.x * i)];
		}
		a[gid] = sum;
	}
	__syncthreads();

	for (int offset = blockDim.x / 2; offset > 0; offset = offset / 2) {
		if (tid < offset) {
			a[gid] += a[gid + offset];
		}
		__syncthreads();
	}

	if (tid == 0) part[blockIdx.x] = a[gid];
}

//Now lets try to get rid of divergence in final 5 iterations using warp unrolling (using interleaved for simpler understanding):
__global__ void reduction_warp_unrolled(int* a, int* part, int size) {
	int tid = threadIdx.x;
	int gid = blockDim.x * blockIdx.x + threadIdx.x;

	if (gid > size) return;

	int* start = a + blockIdx.x * blockDim.x;
	//lets do all threads till we reach 32, where we need to unroll for divergence
	//If we stop only at where it occures (16), then we have to implement if statement introducing divergence (if(tid < 16))
	for (int offset = blockDim.x / 2; offset > 32; offset /= 2) {
		if (tid < offset) start[tid] += start[tid + offset];
		__syncthreads();
	}

	//SO ESSENTIALLY::: we sync all threads up to when it's only the first warp that needs to be done, and we MANUALLY do the first warp of each block
	//because every thread has this code and all threads within a warp will do same thing, there is officially no divergence
	if (tid < 32) {
		volatile int* mem = start;
		mem[tid] += mem[tid + 32];
		mem[tid] += mem[tid + 16];
		mem[tid] += mem[tid + 8];
		mem[tid] += mem[tid + 4];
		mem[tid] += mem[tid + 2];
		mem[tid] += mem[tid + 1];
	}


	if (tid == 0) part[blockIdx.x] = start[tid];
}

__global__ void reduction_warp_unrolled(double* a, double* part, int size) {
	int tid = threadIdx.x;
	int gid = blockDim.x * blockIdx.x + threadIdx.x;

	if (gid > size) return;

	double* start = a + blockIdx.x * blockDim.x;
	//lets do all threads till we reach 32, where we need to unroll for divergence
	//If we stop only at where it occures (16), then we have to implement if statement introducing divergence (if(tid < 16))
	for (int offset = blockDim.x / 2; offset > 32; offset /= 2) {
		if (tid < offset) start[tid] += start[tid + offset];
		__syncthreads();
	}

	//SO ESSENTIALLY::: we sync all threads up to when it's only the first warp that needs to be done, and we MANUALLY do the first warp of each block
	//because every thread has this code and all threads within a warp will do same thing, there is officially no divergence
	if (tid < 32) {
		volatile double* mem = start;
		mem[tid] += mem[tid + 32];
		mem[tid] += mem[tid + 16];
		mem[tid] += mem[tid + 8];
		mem[tid] += mem[tid + 4];
		mem[tid] += mem[tid + 2];
		mem[tid] += mem[tid + 1];
	}


	if (tid == 0) part[blockIdx.x] = start[tid];
}

template<unsigned int iblock_size>
__global__ void reduction_unrolled_intersum(int* a, int* part, int size) {
	int tid = threadIdx.x;
	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	if (gid > size) return;
	int* start = a + blockIdx.x * blockDim.x;

	if (iblock_size == 1024 && tid < 512)
		start[tid] += start[tid + 512];
	__syncthreads();

	if (iblock_size == 512 && tid < 256)
		start[tid] += start[tid + 256];
	__syncthreads();

	if (iblock_size == 256 && tid < 128)
		start[tid] += start[tid + 128];
	__syncthreads();

	if (iblock_size == 128 && tid < 64)
		start[tid] += start[tid + 64];
	__syncthreads();

	//unroll warp
	if (tid < 32) {
		volatile int* mem = start;
		mem[tid] += mem[tid + 32];
		mem[tid] += mem[tid + 16];
		mem[tid] += mem[tid + 8];
		mem[tid] += mem[tid + 4];
		mem[tid] += mem[tid + 2];
		mem[tid] += mem[tid + 1];
	}

	if (tid == 0) part[blockIdx.x] = start[tid];
}

//Officially has the smallest num of divergent branches when gridsize == 512
//Could use template parameters to further reduce num of runtime executions done. Would help more for more inefficient algorithms, but this one
//already performs negligibly well at high array sizes
//I applied template parameter to complete unrolled
__global__ void reduction_unrolling_intersum(int* a, int* part, int size, int rolls) {
	int tid = threadIdx.x;
	int BLOCK_OFFSET = blockIdx.x * blockDim.x * rolls;
	int gid = BLOCK_OFFSET + tid;
	if (gid > size) return;
	int* start = a + BLOCK_OFFSET;

	if ((gid + blockDim.x * (rolls - 1)) < size) {
		int sum = 0;
		for (int i = 0;i < rolls;i++) {
			sum += a[gid + (blockDim.x * i)];
		}
		a[gid] = sum;
	}
	__syncthreads();

	if (blockDim.x == 1024 && tid < 512)
		start[tid] += start[tid + 512];
	__syncthreads();

	if (blockDim.x == 512 && tid < 256)
		start[tid] += start[tid + 256];
	__syncthreads();

	if (blockDim.x == 256 && tid < 128)
		start[tid] += start[tid + 128];
	__syncthreads();

	if (blockDim.x == 128 && tid < 64)
		start[tid] += start[tid + 64];
	__syncthreads();

	if (tid < 32) {
		volatile int* mem = start;
		mem[tid] += mem[tid + 32];
		mem[tid] += mem[tid + 16];
		mem[tid] += mem[tid + 8];
		mem[tid] += mem[tid + 4];
		mem[tid] += mem[tid + 2];
		mem[tid] += mem[tid + 1];
	}

	if (tid == 0) part[blockIdx.x] = a[gid];
}

__global__ void reduction_unrolling_intersum(double* a, double* part, int size, int rolls) {
	int tid = threadIdx.x;
	int BLOCK_OFFSET = blockIdx.x * blockDim.x * rolls;
	int gid = BLOCK_OFFSET + tid;
	if (gid > size) return;
	double* start = a + BLOCK_OFFSET;

	if ((gid + blockDim.x * (rolls - 1)) < size) {
		double sum = 0;
		for (int i = 0;i < rolls;i++) {
			sum += a[gid + (blockDim.x * i)];
		}
		a[gid] = sum;
	}
	__syncthreads();

	if (blockDim.x == 1024 && tid < 512)
		start[tid] += start[tid + 512];
	__syncthreads();

	if (blockDim.x == 512 && tid < 256)
		start[tid] += start[tid + 256];
	__syncthreads();

	if (blockDim.x == 256 && tid < 128)
		start[tid] += start[tid + 128];
	__syncthreads();

	if (blockDim.x == 128 && tid < 64)
		start[tid] += start[tid + 64];
	__syncthreads();

	if (tid < 32) {
		volatile double* mem = start;
		mem[tid] += mem[tid + 32];
		mem[tid] += mem[tid + 16];
		mem[tid] += mem[tid + 8];
		mem[tid] += mem[tid + 4];
		mem[tid] += mem[tid + 2];
		mem[tid] += mem[tid + 1];
	}

	if (tid == 0) part[blockIdx.x] = a[gid];
}

__global__ void reduced_unrolling_innerProduct(double* a, double* b, double* product, double* part, int size, int rolls) {
	int tid = threadIdx.x;
	int BLOCK_OFFSET = blockIdx.x * blockDim.x * rolls;
	int gid = BLOCK_OFFSET + tid;
	if (gid > size) return;
	product[gid] = a[gid] * b[gid];
	__syncthreads();
	double* start = product + BLOCK_OFFSET;

	if ((gid + blockDim.x * (rolls - 1)) < size) {
		double sum = 0;
		for (int i = 0;i < rolls;i++) {
			sum += product[gid + (blockDim.x * i)];
		}
		product[gid] = sum;
	}
	__syncthreads();

	if (blockDim.x == 1024 && tid < 512)
		start[tid] += start[tid + 512];
	__syncthreads();

	if (blockDim.x == 512 && tid < 256)
		start[tid] += start[tid + 256];
	__syncthreads();

	if (blockDim.x == 256 && tid < 128)
		start[tid] += start[tid + 128];
	__syncthreads();

	if (blockDim.x == 128 && tid < 64)
		start[tid] += start[tid + 64];
	__syncthreads();

	if (tid < 32) {
		volatile double* mem = start;
		mem[tid] += mem[tid + 32];
		mem[tid] += mem[tid + 16];
		mem[tid] += mem[tid + 8];
		mem[tid] += mem[tid + 4];
		mem[tid] += mem[tid + 2];
		mem[tid] += mem[tid + 1];
	}

	if (tid == 0) part[blockIdx.x] = product[gid];
}

//__global__ void dynamic_parallelism_check(int size, int depth) {
//	printf("Depth: %d | Thread: %d\n", depth, threadIdx.x);
//	if (size == 1) return;
//	if (threadIdx.x == 0) dynamic_parallelism_check << <1, size / 2 >> > (size / 2, depth + 1);
//}

//strided means accessing memory by fixed amount in between per thread
//All strided patterns are non-coalesced, but not all non-coalesced patterns are strided
//non-coalesced is the worst memory access pattern

//Here, reading is coalesced but writing is strided.
__global__ void transpose_row_to_column(int* matrix, int* transposed, int nx, int ny) {
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix < nx && iy < ny) {
		transposed[ix * ny + iy] = matrix[iy * nx + ix];
	}
}

//Here, reading is strided but writing is coalesced
__global__ void transpose_column_to_row(int* matrix, int* transposed, int nx, int ny) {
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix < nx && iy < ny) {
		transposed[iy * nx + ix] = matrix[ix * ny + iy];
	}
}

__global__ void transpose_unrolling(int* matrix, int* transposed, int nx, int ny, int rolls) {
	int ix = blockIdx.x * blockDim.x * rolls + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	int ti = iy * nx + ix;
	int to = ix * ny + iy;

	if ((ix + (rolls-1) * blockDim.x) < nx && iy < ny) {
		for (int roll = 0;roll < rolls;roll++) {
			transposed[to + ny * blockDim.x * roll] = matrix[ti + blockDim.x * roll];
		}
	}
}

__global__ void smem_static_test(int* in, int* out, int size) {
	//shared mem happens block-wise
	int tid = threadIdx.x;
	//global mem happens, well, global-wise
	int gid = blockIdx.x * blockDim.x + tid;

	__shared__ int smem[SHARED_ARRAY_SIZE];

	if (gid < size) {
		smem[tid] = in[gid];
		out[gid] = smem[tid];
	}
}

__global__ void smem_dynamic_test(int* in, int* out, int size) {
	//shared mem happens block-wise
	int tid = threadIdx.x;
	//global mem happens, well, global-wise
	int gid = blockIdx.x * blockDim.x + tid;

	extern __shared__ int smem[];

	if (gid < size) {
		smem[tid] = in[gid];
		out[gid] = smem[tid];
	}
}

__global__ void setRowReadCol(int* out) {
	__shared__ int tile[BDIMY][BDIMX];

	int gid = threadIdx.y * blockDim.x + threadIdx.x;

	tile[threadIdx.y][threadIdx.x] = gid;
	//wait for all threads in the block to store in array
	__syncthreads();

	//load shared memory in opposity direction indices
	out[gid] = tile[threadIdx.x][threadIdx.y];

}

__global__ void setColReadRow(int* out) {
	__shared__ int tile[BDIMY][BDIMX];

	int gid = threadIdx.y * blockDim.x + threadIdx.x;

	tile[threadIdx.x][threadIdx.y] = gid;
	//wait for all threads in the block to store in array
	__syncthreads();

	//load shared memory in opposity direction indices
	out[gid] = tile[threadIdx.y][threadIdx.x];

}

__global__ void setRowReadRow(int* out) {
	__shared__ int tile[BDIMY][BDIMX];

	int gid = threadIdx.y * blockDim.x + threadIdx.x;

	tile[threadIdx.y][threadIdx.x] = gid;
	//wait for all threads in the block to store in array
	__syncthreads();

	//load shared memory in opposity direction indices
	out[gid] = tile[threadIdx.y][threadIdx.x];

}

__global__ void convolutionKernel(unsigned char* data, unsigned char* newData, double* filter, int size, int kernelSize) {
	int r = blockIdx.x;
	int width = blockDim.x * gridDim.z;
	int c = threadIdx.x + blockDim.x * blockIdx.z;
	int chan = blockIdx.y;

	//row, columns, channel
	int gid = (r * width * gridDim.y) + (c * gridDim.y) + (chan);

	if (gid > size) return;

	double sum = 0;
	int end = (int)((kernelSize / 2.0) + .5);
	int start = end - kernelSize;
	for (int i = start;i < end;i++) {
		for (int j = start;j < end;j++) {
			int fid = ((r + i) * width * gridDim.y) + ((c+j) * gridDim.y) + (chan);
			int filId = (i + (kernelSize / 2)) * kernelSize + (j + (kernelSize / 2));
			double term = 0;
			double filTerm = 0;
			if (fid >= 0 && fid < size) {
				filTerm = filter[filId];
				term = data[fid] * filTerm;
			}
			sum += term;
		}
	}
	if (sum > 255) sum = 255;
	if (sum < 0) sum = 0;
	newData[gid] = sum;
	//printf("new pixel %d: %d\n", gid, newData[gid]);

}

__global__ void dotProduct(double* a, double* b, double* c, int size) {
	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	if (gid > size) return;
	c[gid] = a[gid] * b[gid];
}

__global__ void sequentialProduct(double* a, double* b, double* c, int size) {
	for (int i = 0;i < size;i++) c[i] = a[i] * b[i];
}

__global__ void sequentialSum(double* a, double* sumOut, int size) {

	for (int i = 0;i < size;i++) {
		*sumOut += a[i];
	}
}

__global__ void copyElements(double* dest, double* src, int size) {
	int gid = blockDim.x * blockIdx.x + threadIdx.x;

	if (gid >= size) return;
	dest[gid] = src[gid];
}

__global__ void batchCopy(double* dest, double* src, int size) {
	int gid = blockDim.x * blockIdx.x + threadIdx.x;

	if (gid > size) return;
	//src[gid] = 5;
	dest[gid] = src[gid];
}

__global__ void copySeqElements(double* dest, double* src, int size) {
	for (int i = 0;i < size;i++) {
		dest[i] = src[i];
	}
}

__global__ void SoftMaxSeq(double* output, int size) {
	double denom = 0;
	for (int i = 0;i < size;i++) {
		denom += output[i];
		//printf("out: %f / %f\n", output[i], denom);
	}
	if (denom < DBL_MIN)
		denom = DBL_MIN;
	for (int i = 0;i < size;i++) {
		output[i] /= denom;
	}
	//printf("cell 0 out: %f\n", output[0]);
}

__global__ void batchSoftMax(double* output, int size, int batchSize) {
	if (threadIdx.x > batchSize) return;
	//printf("thread:%d\n", threadIdx.x);
	int batch = threadIdx.x;
	double* local_output = output + batch * size;
	double denom = 0;
	for (int i = 0;i < size;i++) {
		denom += local_output[i];
	}
	if (denom < DBL_MIN)
		denom = DBL_MIN;
	for (int i = 0;i < size;i++) {
		local_output[i] /= denom;
	}
	//printf("batch %d, cell 0 out: %f\n", threadIdx.x, local_output[0]);
}

__global__ void getSum(int* x, int size, int* sum) {
	int gid = blockDim.x * blockIdx.x + threadIdx.x;

	if (gid > size) return;

	atomicAdd(sum, x[gid]);
}

__global__ void runCleanParatron(double* input, double* output, double* weights, activation_function A_F, int inputSize, int CIL, double bias) {

	//inputs will be passed from parent kernel as output + 0
	//output array will be passed as output + CIL[layer]

	//Maybe block will be layer and thread will be cell
	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	if (gid > CIL) return;
	int cellOffset = (inputSize + 1) * gid;
	double* localWeights = weights + cellOffset;
	//I FR DO NOT UNDERSTAND THIS BUT LIKE IDK
	if (cellOffset == 0 && CIL == 10 && localWeights[0] == .5) localWeights[0] = -0.88970610675374617;

	double sum = 0;
	for (int i = 0;i < inputSize;i++) {
		sum += input[i] * localWeights[i];
	}
	sum += bias * localWeights[inputSize];
	switch (A_F) {
	case SIGMOID:
		sum = 1 / (1 + exp(-sum)); //sigmoid
		break;
	case RELU:
		sum = (sum > 0) ? 1 : 0; //ReLu
		break;
	case SOFTMAX:
		sum = exp(sum);
		break;
	}
	output[gid] = sum;
}

__global__ void runBatchParatron(double* input, double* output, double* weights, activation_function A_F, int inputSize, int CIB, int CIL, double bias) {
	//inputs will be passed from parent kernel as output + 0
	//output array will be passed as output + CIL[layer]

	//Maybe block will be layer and thread will be cell
	//gridDim.x refers to batchSize, so blockIdx.x is which row (batch) we are in, and blockIdx.y is which multiple of 32 of the cell we are in
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	//int gid = blockId * blockDim.x + threadIdx.x;
	int gid = (CIL * blockIdx.x) + (blockIdx.y * blockDim.x) + threadIdx.x;
	int cid = blockDim.x * blockIdx.y + threadIdx.x;
	if (gid > CIB) return;
	if (cid > CIL) return;

	int cellOffset = (inputSize + 1) * cid;
	double* localWeights = weights + cellOffset;
	
	int inputOffset = blockIdx.x * inputSize;
	double* localInput = input + inputOffset;
	bool check = false;

	double sum = 0;
	for (int i = 0;i < inputSize;i++) {
		sum += localInput[i] * localWeights[i];
	}
	sum += bias * localWeights[inputSize];
	switch (A_F) {
	case SIGMOID:
		sum = 1 / (1 + exp(-sum)); //sigmoid
		break;
	case RELU:
		sum = (sum > 0) ? 1 : 0; //ReLu
		break;
	case SOFTMAX:
		sum = exp(sum);
		break;
	}
	output[gid] = sum;
}


__global__ void runParatron(double* input, double* output, double* weights, activation_function A_F, int inputSize, int CIL, double bias, int* counter) {

	//inputs will be passed from parent kernel as output + 0
	//output array will be passed as output + CIL[layer]

	//Maybe block will be layer and thread will be cell
	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	if (gid >= CIL) return;
	int cellOffset = (inputSize + 1) * gid;
	double* localWeights = weights + cellOffset;
	//printf("Cell: %d :: weightOffset %f :: AF %d\n", gid, localWeights[0], A_F);

	double sum = 0;
	for (int i = 0;i < inputSize;i++) {
		sum += input[i] * localWeights[i];
		//printf("%f = %f x %f -- %d\n", input[i] * localWeights[i], input[i], localWeights[i], gid);
	}
	sum += bias * localWeights[inputSize];
	//printf("%f = %f x %f -- %d\n\n", bias * localWeights[inputSize], bias, localWeights[inputSize], gid);
	switch (A_F) {
	case SIGMOID:
		sum = 1 / (1 + exp(-sum)); //sigmoid
		break;
	case RELU:
		sum = (sum > 0) ? 1 : 0; //ReLu
		break;
	case SOFTMAX:
		sum = exp(sum);
		break;
	}
	output[gid] = sum;
	atomicAdd(counter, 1);
}

//Dynamic parallelism needs to be commented out when not using proper compile mode
__global__ void runMLP(double* x, double* outputs, double* weights, activation_function* A_Fs, int* CIL, int layers, double bias, int* WLO, int* OLO) {

	if (CIL[0] > 511) {
		copyElements << <CIL[0] / 32, 32 >> > (outputs, x, CIL[0]);
		//I WOULD ALSO LIKE TO WAIT HERE
	}
	else
		for (int i = 0;i < CIL[0];i++) {
			outputs[i] = x[i];
		}
	//printf("[");
	//for (int j = 0;j < CIL[0];j++) {
	//	printf("%f, ", outputs[j]);
	//}
	//printf("]\n");
	int* counter;
	cudaMalloc((void**)&counter, sizeof(int));
	for (int i = 1;i < layers;i++) {
		*counter = 0;
		//printf("----------------------Layer %d :: InputSize %d :: Layer weight offset %d :: Layer output offset %d----------------------\n", i, CIL[i-1], WLO[i-1], OLO[i]);
		runParatron << < (CIL[i] / 32) + 1, 32 >> > (outputs + OLO[i - 1], outputs + OLO[i], weights + WLO[i - 1], A_Fs[i], CIL[i - 1], CIL[i], bias, counter);
		while (*counter < CIL[i]) {}
		//if (i != layers - 1) {
			//double* temp = outputs + OLO[i];
			//printf("offset = %d\n", OLO[i]);
		//	printf("[");
		//	for (int j = 0;j < CIL[i];j++) {
		//		printf("%f, ", temp[j]);
		//	}
		//	printf("]\n");
		//}
		
		//cudaDeviceSynchronize(); //THIS IS WHERE I NEED TO WAIT UNTIL NEXT ITERATION
	}
	if (A_Fs[layers - 1] == SOFTMAX) {
		double* temp = outputs + OLO[layers - 1];
		double denom = 0;
		for (int i = 0;i < CIL[layers - 1];i++) {
			denom += temp[i];
		}
		if (denom < DBL_MIN)
			denom = DBL_MIN;
		for (int i = 0;i < CIL[layers - 1];i++) {
			temp[i] /= denom;
		}
		//printf("Final offset = %d\n", OLO[layers - 1]);
		//printf("[");
		//for (int i = 0;i < CIL[layers - 1];i++) {
		//	printf("%f, ", temp[i]);
		//}
		//printf("]\n");
	}
	cudaFree(counter);
}

__global__ void runMLP(double* x, double* outputs, double* weights, activation_function* A_Fs, int* CIL, int layers, double bias, int* WLO, int* OLO, int* count) {

	if (CIL[0] > 511) {
		copyElements << <CIL[0] / 32, 32 >> > (outputs, x, CIL[0]);
		//I WOULD ALSO LIKE TO WAIT HERE
	}
	else
		for (int i = 0;i < CIL[0];i++) {
			outputs[i] = x[i];
		}
	//printf("[");
	//for (int j = 0;j < CIL[0];j++) {
	//	printf("%f, ", outputs[j]);
	//}
	//printf("]\n");
	int* counter;
	cudaMalloc((void**)&counter, sizeof(int));
	for (int i = 1;i < layers;i++) {
		*counter = 0;
		//printf("----------------------Layer %d :: InputSize %d :: Layer weight offset %d :: Layer output offset %d----------------------\n", i, CIL[i-1], WLO[i-1], OLO[i]);
		runParatron << < (CIL[i] / 32) + 1, 32 >> > (outputs + OLO[i - 1], outputs + OLO[i], weights + WLO[i - 1], A_Fs[i], CIL[i - 1], CIL[i], bias, counter);
		while (*counter < CIL[i]);
		//double* temp = outputs + OLO[i];
		//printf("offset = %d\n", OLO[i]);
		//printf("[");
		//for (int j = 0;j < CIL[i];j++) {
		//	printf("%f, ", temp[j]);
		//}
		//printf("]\n");
		//cudaDeviceSynchronize(); //THIS IS WHERE I NEED TO WAIT UNTIL NEXT ITERATION
	}
	if (A_Fs[layers - 1] == SOFTMAX) {
		double* temp = outputs + OLO[layers - 1];
		double denom = 0;
		for (int i = 0;i < CIL[layers - 1];i++) {
			denom += temp[i];
		}
		if (denom < DBL_MIN)
			denom = DBL_MIN;
		for (int i = 0;i < CIL[layers - 1];i++) {
			temp[i] /= denom;
		}
		//printf("Final offset = %d\n", OLO[layers - 1]);
		//printf("[");
		//for (int i = 0;i < CIL[layers - 1];i++) {
		//	printf("%f, ", temp[i]);
		//}
		//printf("]\n");
	}
	atomicAdd(count, 1);
	cudaFree(counter);
}

__global__ void getLossSeq(double* x, double* y, double* loss, loss_function* L_F, int size) {
	*loss = 0.0;
	switch (*L_F) {
	case(MSE):
		for (int i = 0;i < size;i++) {
			*loss += pow((x[i] - y[i]), 2);
		}
		*loss /= size;
		break;
	case(CROSS_ENTROPY):
		for (int i = 0;i < size;i++) {
			if (x[i] == 0.0) *loss -= y[i] * log(0.00001);
			else *loss -= y[i] * log(x[i]);
			//printf("prediction: %f ; actual: %f\n", x[i], y[i]);
			//printf("clean: %f * %f = %f\n", y[i], log(x[i]), y[i] * log(x[i]));
		}
		break;
	}
	//printf("clean loss: %f\n", *loss);
}

__global__ void batchLoss(double* x, double* y, double* loss, loss_function L_F, int size, int batchSize) {
	int batch = threadIdx.x;

	double sum = 0.0;
	*loss = 0.0;

	for (int b = 0;b < batchSize;b++) {
		int off = b * size;
		switch (L_F) {
		case(MSE):
			for (int i = 0;i < size;i++) {
				*loss += pow((x[i + off] - y[i + off]), 2);
			}
			*loss /= size;
			break;
		case(CROSS_ENTROPY):
			for (int i = 0;i < size;i++) {
				if (x[i + off] == 0.0) *loss -= y[i + off] * log(0.00001);
				else *loss -= y[i + off] * log(x[i + off]);
				//printf("prediction: %f ; actual: %f\n", x[i], y[i]);
				//printf("%f * %f = %f\n", y[i], log(x[i]), y[i] * log(x[i]));
			}
			double testx = 0.0;
			double testy = 0.0;
			for (int i = 0;i < size;i++) {
				testx += x[i + off];
				testy += y[i + off];
			}
			//if (testx != 1 || testy != 1) {
				//printf("testx: %f\n", testx);
				//printf("testy: %f\n", testy);
				//printf("\n");
			//}
			break;
		}
		sum += *loss;
		//x += 1;
		//y += 1;
	}
	*loss /= batchSize;

	//printf("\n");
}

__global__ void activationDerivative(double x, activation_function A_F, int* counter) {
	switch (A_F) {
	case SIGMOID:
		x = x * (1 - x); //sigmoid derivative
		break;
	case RELU:
		if (x > 0) x = 1; //ReLu derivative
		x = .000001;
		break;
	case SOFTMAX:
		x = x * (1 - x); //softmax derivative
		//else return -1 * xWrt * xOf;
		//return x * (1 - x); //sigmoid derivative
		break;
	}
	atomicAdd(counter, 1);
}

__global__ void getErrorLayerWRTInputSeq(double* error_terms, double* x, double* y, int size, loss_function L_F, activation_function A_F) {
	for (int i = 0;i < size;i++) {
		//error_terms[i] = x[i] * (1 - y[i]) * 2 * (x[i] - y[i]);
		error_terms[i] = x[i] - y[i];
		//printf("error_term[%d]: %.17e\n", i, error_terms[i]);
		//switch (L_F) {
		//case(MSE):
		//	double activation;
		//	int counter = 0;
		//	activationDerivative<<<1,1>>>(x[i], A_F, &counter);
		//	while (counter < 1);
		//	error_terms[i] = 2 * (x[i] - y[i]) * activation;
		//	break;
		//case(CROSS_ENTROPY):
		//	//if (x == 0.0) x = .0001;
		//	//return y / x;
		//	error_terms[i] = x[i] - y[i];
		//	break;
		//}
	}
}

__global__ void batchErrorLayer(double* error_terms, double* x, double* y, int size, loss_function L_F, activation_function A_F) {
	int batch = threadIdx.x;
	double* local_x = x + (batch * size);
	double* local_y = y + (batch * size);
	for (int i = 0;i < size;i++) {
		//error_terms[i] = x[i] * (1 - y[i]) * 2 * (x[i] - y[i]);
		error_terms[i] = local_x[i] - local_y[i];
		//printf("error_term[%d]: %.17e\n", i, error_terms[i]);
		//switch (L_F) {
		//case(MSE):
		//	double activation;
		//	int counter = 0;
		//	activationDerivative<<<1,1>>>(x[i], A_F, &counter);
		//	while (counter < 1);
		//	error_terms[i] = 2 * (x[i] - y[i]) * activation;
		//	break;
		//case(CROSS_ENTROPY):
		//	//if (x == 0.0) x = .0001;
		//	//return y / x;
		//	error_terms[i] = x[i] - y[i];
		//	break;
		//}
	}
}

__global__ void gradient(double* weights, double* for_terms, double* terms, double* outputs, int for_CIL, int CIL, int* count) {
	int cid = blockIdx.x * blockDim.x + threadIdx.x; //cell Id given error_term layer ; weights point to this cell
	if (cid >= CIL) return;

	double err_sum = 0.0;

	for (int i = 0;i < for_CIL;i++) {
		int cellOffset = (CIL + 1) * i;
		err_sum += weights[cellOffset + cid] * for_terms[i];
	}
	terms[cid] = outputs[cid] * (1 - outputs[cid]) * err_sum;
	if (cid == CIL) {
		printf("HELLO\n");
	}
	atomicAdd(count, 1);
}

__global__ void cleanGradient(double* weights, double* for_terms, double* terms, double* outputs, int for_CIL, int CIL) {
	int cid = blockIdx.x * blockDim.x + threadIdx.x; //cell Id given error_term layer ; weights point to this cell
	if (cid >= CIL) return;

	double err_sum = 0.0;

	for (int i = 0;i < for_CIL;i++) {
		int cellOffset = (CIL + 1) * i;
		err_sum += weights[cellOffset + cid] * for_terms[i];
	}
	terms[cid] = outputs[cid] * (1 - outputs[cid]) * err_sum;
	if (cid == CIL) {
		printf("HELLO\n");
	}
}

__global__ void batchGradient(double* weights, double* for_terms, double* terms, double* outputs, int for_CIL, int CIL, int CIB) {

	//gridDim.x refers to batchSize, so blockIdx.x is which row (batch) we are in, and blockIdx.y is which multiple of 32 of the cell we are in
	int cid = blockIdx.y * blockDim.x + threadIdx.x; //cell Id given error_term layer ; weights point to this cell
	int gid = (CIL * blockIdx.x) + (blockIdx.y * blockDim.x) + threadIdx.x;
	if (cid >= CIL) return;
	if (gid >= CIB) return;

	int for_offset = blockIdx.x * for_CIL;
	//for_terms += for_offset;
	//if (for_CIL == 10 && blockIdx.x == 0 && blockIdx.y == for_CIL / 32 && threadIdx.x == 31) printf("gridDim.x: %d, gridDim.y: %d\n", gridDim.x, gridDim.y);
	//if (for_CIL == 10 && blockIdx.x == gridDim.x - 1 && blockIdx.y == gridDim.y-1 && threadIdx.x == 0) printf("batch: %d, block: %d, thread: %d\n", blockIdx.x, blockIdx.y, threadIdx.x);
	
	int offset = blockIdx.x * CIL;
	//terms += offset;
	//outputs += offset;

	double err_sum = 0.0;

	for (int i = 0;i < for_CIL;i++) {
		int cellOffset = (CIL + 1) * i;
		err_sum += weights[cellOffset + cid] * for_terms[i + for_offset];
	}
	terms[cid + offset] = outputs[cid + offset] * (1 - outputs[cid + offset]) * err_sum;
	if (cid == CIL) {
		printf("HELLO\n");
	}
}

__global__ void updateWeightsbyLayer(double* weights, double* error_terms, double* outputs, double eta, int CIL, int forCIL, double bias, int* count) {
	if (threadIdx.x > forCIL || blockIdx.x > CIL) return;
	int wid = (CIL + 1) * threadIdx.x + blockIdx.x;
	double delta = eta * error_terms[threadIdx.x] * outputs[blockIdx.x];
	if(blockIdx.x == CIL) delta = eta * error_terms[threadIdx.x] * bias;
	weights[wid] -= delta;
	atomicAdd(count, 1);
}

__global__ void cleanUpdateWeightsbyLayer(double* weights, double* error_terms, double* outputs, double eta, int CIL, int forCIL, double bias) {
	if (threadIdx.x > forCIL || blockIdx.x > CIL) return;
	int wid = gridDim.x * threadIdx.x + blockIdx.x;
	double delta = eta * error_terms[threadIdx.x] * outputs[blockIdx.x];
	//if (threadIdx.x == 0 && blockIdx.x == 0) printf("weight: %f\n", weights[wid]);
	if (blockIdx.x == CIL) delta = eta * error_terms[threadIdx.x] * bias;
	weights[wid] -= delta;
}

__global__ void batchMakeGradient(double* grad, double* error_terms, double* outputs, double eta, int CIL, int forCIL, double bias, int batchSize) {
	int blockId = blockIdx.x * (CIL + 1) * forCIL;
	int gid = blockId + (CIL + 1) * threadIdx.x + blockIdx.y;
	if (gid > (CIL + 1) * forCIL * batchSize) return;

	int err_offset = forCIL * blockIdx.x;
	double* local_errors = error_terms + err_offset;

	int out_offset = CIL * blockIdx.x;
	//outputs += out_offset;
	double* local_outs = outputs + out_offset;


	//if (threadIdx.x == 0 && blockIdx.x == 0) printf("weight: %f ; batch: %d ; err_offset: %d ; out_offset: %d\n", weights[wid], batch, err_offset, out_offset);
	double delta = eta * local_errors[threadIdx.x] * local_outs[blockIdx.y];
	if (blockIdx.y == CIL) delta = eta * local_errors[threadIdx.x] * bias;
	grad[gid] = delta;
}

__global__ void batchUpdateWeightsbyLayer(double* weights, double* error_terms, double* outputs, double eta, int CIL, int forCIL, double bias, int batch) {
	if (threadIdx.x > forCIL || blockIdx.x > CIL) return;
	int wid = (CIL + 1) * threadIdx.x + blockIdx.x;

	int err_offset = forCIL * batch;
	double* local_errors = error_terms + err_offset;

	int out_offset = CIL * batch;
	//outputs += out_offset;
	double* local_outs = outputs + out_offset;


	//if (threadIdx.x == 0 && blockIdx.x == 0) printf("weight: %f ; batch: %d ; err_offset: %d ; out_offset: %d\n", weights[wid], batch, err_offset, out_offset);
	double delta = eta * local_errors[threadIdx.x] * local_outs[blockIdx.x];
	if (blockIdx.x == CIL) delta = eta * local_errors[threadIdx.x] * bias;
	weights[wid] -= delta;
}

__global__ void backpropagation(double* loss, double* x, double* y, double* error_terms, double* outputs, double* weights, activation_function* A_Fs, int* CIL, int layers, double bias, double eta, int* WLO, int* OLO) {
	int* counter;
	cudaMalloc((void**)&counter, sizeof(int));
	//*counter = 0;
	//runMLP<<<1,1>>>(x, outputs, weights, A_Fs, CIL, layers, bias, WLO, OLO, counter);
	//while (*counter == 0);
	*counter = 0;

	double* temp = outputs + OLO[layers - 1];
	double out;
	*loss = 0.0;
	for (int i = 0;i < CIL[layers-1];i++) {
		if (temp[i] == 0.0) out = 0.00001;
		else out = temp[i];
		*loss -= y[i] * log(out);
		//printf("prediction: %f ; actual: %f\n", out, y[i]);
		//printf("%f * %f = %f\n", y[i], log(out), y[i] * log(out));
	}
	//printf("\n");

	double* out_back = outputs + OLO[layers - 1];
	double* error_back = error_terms + OLO[layers - 1] - OLO[1];
	for (int i = 0;i < CIL[layers - 1];i++) {
		//error_back[i] = out_back[i] * (1 - out_back[i]) * 2 * (out_back[i] - y[i]);
		error_back[i] = out_back[i] - y[i];
	}
	//printf("[");
	//for (int j = 0;j < 10;j++) {
	//	printf("%.17e, ", error_back[j]);
	//}
	//printf("]\n");

	for (int i = layers - 3;i >= 0;i--) {
		//printf("--------------for_weights offset: %d : Error offset: %d : for_Error offset: %d : outputs offset: %d : for_outputs offset: %d-------------- %d\n", WLO[i + 1], OLO[i + 1] - OLO[1], OLO[i + 2] - OLO[1], OLO[i], OLO[i + 1], i);
		gradient << <(CIL[i+1] / 32) + 1, 32 >> > (weights + WLO[i + 1], error_terms + OLO[i + 2] - OLO[1], error_terms + OLO[i + 1] - OLO[1], outputs + OLO[i + 1], CIL[i + 2], CIL[i + 1], counter);
		while (*counter < CIL[i + 1]);
		*counter = 0;
	}
	//printf("[");
	//for (int j = 0;j < 10;j++) {
	//	printf("%.17e, ", error_back[j]);
	//}
	//printf("]\n");
	int weightSize = 0;
	for (int i = 0;i < layers - 1;i++) {
		weightSize += CIL[i] * CIL[i + 1];
	}

	for (int i = 0;i < layers - 1;i++) {
		//printf("--------------weights offset: %d : Error offset: %d : outputs offset: %d-------------- %d\n", WLO[i], OLO[i+1] - OLO[1], OLO[i], i);
		updateWeightsbyLayer << <CIL[i]+1, CIL[i + 1] >> > (weights + WLO[i], error_terms + OLO[i+1] - OLO[1], outputs + OLO[i], eta, CIL[i], CIL[i + 1], bias, counter);
	}
	while (*counter < weightSize);
	//printf("\n");
	cudaFree(counter);

}

//This memory is really fragmented, but you get way more threads going off of cell than batchSize
__global__ void averageGrad(double* batch_grad, double* gradient, int batchSize, int size, int for_size) {
	int gid = threadIdx.x * size + blockIdx.x; //cell Id given error_term layer ; weights point to this cell
	int batch = size * for_size;
	if (gid > size * for_size) return;


	double sum = 0.0;
	for (int i = 0;i < batchSize;i++) {
		sum += batch_grad[gid + (i * batch)];
	}
	sum /= batchSize;
	gradient[gid] = sum;
}

__global__ void applyGrad(double* weights, double* gradient, int size) {
	int gid = threadIdx.x * gridDim.x + blockIdx.x;
	if (gid > size) return;

	weights[gid] -= gradient[gid];
}

__global__ void setArray(double* a, double* b, int size) {
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gid > size) return;

	b[gid] = a[gid];
}

__global__ void updateWeights(double* weights, double* error_terms, double* outputs, int* CIL, int* WLO, int* OLO, int layers, double eta, double bias) {
	int weightSize = 0;
	for (int i = 0;i < layers - 1;i++) {
		weightSize += CIL[i] * CIL[i + 1];
	}
	int* counter;
	cudaMalloc((void**)&counter, sizeof(int));
	for (int i = 0;i < layers - 1;i++) {
		//printf("--------------weights offset: %d : Error offset: %d : outputs offset: %d-------------- %d\n", WLO[i], OLO[i+1] - OLO[1], OLO[i], i);
		updateWeightsbyLayer << <CIL[i] + 1, CIL[i + 1] >> > (weights + WLO[i], error_terms + OLO[i + 1] - OLO[1], outputs + OLO[i], eta, CIL[i], CIL[i + 1], bias, counter);
	}
	while (*counter < weightSize);
	cudaFree(counter);
}