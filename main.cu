#include "kernels.cuh"
#include "general.h"

int main() {

	int size = 2 << 27;
	int bSize = size * sizeof(int);
	int blockSize = 128;
	int partSize = sizeof(int) * size / blockSize;
	printf("Array Size: %d, Partition Size: %d\n", size, partSize / sizeof(int));


	int* h_a = (int*)malloc(bSize);
	int* h_part = (int*)malloc(partSize);

	generateRandArray(h_a, 0x5, size);
	//printArray(h_a, size);

	clock_t cpu_start, cpu_end;
	cpu_start = clock();
	int cpuSum = getSum(h_a, size);
	cpu_end = clock();

	dim3 block(blockSize);
	dim3 grid(size / block.x);
	printf("Kernel launch parameters | grid.x: %d, block.x: %d\n", grid.x, block.x);

	int* d_a, *d_part;

	gpuErrorchk(cudaMalloc((void**)&d_a, bSize));
	gpuErrorchk(cudaMalloc((void**)&d_part, partSize));
	gpuErrorchk(cudaMemset(d_part, 0, partSize));


	clock_t htod_start, htod_end;
	htod_start = clock();
	gpuErrorchk(cudaMemcpy(d_a, h_a, bSize, cudaMemcpyHostToDevice));
	htod_end = clock();

	clock_t gpu_start, gpu_end;
	gpu_start = clock();
	neighbored_paired_reduction << <grid,block >> > (d_a, d_part, size);
	gpuErrorchk(cudaDeviceSynchronize());
	gpu_end = clock();
	
	gpuErrorchk(cudaMemcpy(d_a, h_a, bSize, cudaMemcpyHostToDevice));
	gpuErrorchk(cudaMemset(d_part, 0, partSize));

	int rollCount = grid.x / 512;
	clock_t inter_start, inter_end;
	inter_start = clock();
	reduction_unrolled_intersum <128> << <grid, block >> > (d_a, d_part, size);
	gpuErrorchk(cudaDeviceSynchronize());
	inter_end = clock();

	gpuErrorchk(cudaMemcpy(d_a, h_a, bSize, cudaMemcpyHostToDevice));
	gpuErrorchk(cudaMemset(d_part, 0, partSize));

	rollCount = 2048;
	clock_t reduced_start, reduced_end;
	reduced_start = clock();
	reduction_unrolling_intersum << <512, block >> > (d_a, d_part, size, rollCount);
	gpuErrorchk(cudaDeviceSynchronize());
	reduced_end = clock();
	
	int gpu = 0;
	clock_t dtoh_start, dtoh_end;
	dtoh_start = clock();
	gpuErrorchk(cudaMemcpy(h_part, d_part, partSize, cudaMemcpyDeviceToHost));
	//printArray(h_part, grid.x);
	for (int i = 0;i < (grid.x);i++) {
		gpu += h_part[i];
	}
	dtoh_end = clock();

	compare_arrays(&cpuSum, &gpu, 1);

	printExecution("htod", htod_start, htod_end);
	printExecution("dtoh", dtoh_start, dtoh_end);
	printExecution("CPU", cpu_start, cpu_end);
	printExecution("GPU", gpu_start, gpu_end);
	printExecution("GPU EFFICIENT", reduced_start, reduced_end);
	printExecution("GPU WARP UNROLLED", inter_start, inter_end);
	printExecution("Sum of GPU tasks", gpu_start + dtoh_start + htod_start, gpu_end + dtoh_end + htod_end);
	printExecution("Sum of GPU efficient", reduced_start + dtoh_start + htod_start, reduced_end + dtoh_end + htod_end);


	cudaFree(d_a);
	cudaDeviceReset();



	return 0;
}