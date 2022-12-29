#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <time.h>
#include <math.h>

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


