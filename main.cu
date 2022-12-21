#include "kernels.cuh"
#include "general.h"

int main() {

	int size = 1 << 22;
	//int bSize = size * sizeof(int);
	int blockSize = 128;

	//int* h_a = (int*)malloc(bSize);

	clock_t cpu_start, cpu_end;
	//cpu_start = clock();
	//cpu_end = clock();



	//int* d_a;
	//gpuErrorchk(cudaMalloc((void**)&d_a, bSize));

	//clock_t htod_start, htod_end;
	//htod_start = clock();
	//gpuErrorchk(cudaMemcpy(d_a, h_a, bSize, cudaMemcpyHostToDevice));
	//htod_end = clock();

	dim3 block(blockSize);
	dim3 grid((size + blockSize - 1) / blockSize);

	//clock_t gpu_start, gpu_end;
	//gpu_start = clock();
	no_divergence << <grid,block >> > ();
	gpuErrorchk(cudaDeviceSynchronize());
	divergent << <grid, block >> > ();
	gpuErrorchk(cudaDeviceSynchronize());
	//gpu_end = clock();

	//clock_t dtoh_start, dtoh_end;
	//dtoh_start = clock();
	//gpuErrorchk(cudaMemcpy(h_d, d_d, bSize, cudaMemcpyDeviceToHost));
	//dtoh_end = clock();
	//cudaFree(&d_a);

	cudaDeviceReset();

	//printf("htod: %4.15f\n", (double)((double)(htod_end - htod_start) / CLOCKS_PER_SEC));
	//printf("dtoh: %4.15f\n", (double)((double)(dtoh_end - dtoh_start) / CLOCKS_PER_SEC));
	//printf("CPU: %4.15f\n", (double)((double)(cpu_end - cpu_start) / CLOCKS_PER_SEC));
	//printf("GPU: %4.15f\n", (double)((double)(gpu_end - gpu_start) / CLOCKS_PER_SEC));
	//printf("Sum of GPU tasks: %4.15f\n", (double)(((double)(htod_end - htod_start) / CLOCKS_PER_SEC) + ((double)(dtoh_end - dtoh_start) / CLOCKS_PER_SEC) + ((double)(gpu_end - gpu_start) / CLOCKS_PER_SEC)));




	return 0;
}