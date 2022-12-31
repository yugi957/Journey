#include "kernels.cuh"
#include "general.h"
#include "general.cuh"

void printData(char* msg, int* in, const int size)
{
	printf("%s: ", msg);

	for (int i = 0; i < size; i++)
	{
		printf("%5d", in[i]);
		fflush(stdout);
	}

	printf("\n");
	return;
}


void pinnedMemSumExample(int size) {

	int bSize = size * sizeof(int);
	int blockSize = 128;
	int partSize = sizeof(int) * size / blockSize;
	printf("Array Size: %d, Partition Size: %d\n", size, partSize / sizeof(int));


	int* h_a = (int*)malloc(bSize);
	int* h_part = (int*)malloc(partSize);
	int* pinned_h_a;
	gpuErrorchk(cudaMallocHost(&pinned_h_a, bSize));
	int* pinned_h_part;
	gpuErrorchk(cudaMallocHost(&pinned_h_part, partSize));

	time_t t;
	srand((unsigned)time(&t));
	for (int i = 0;i < size;i++) {
		h_a[i] = pinned_h_a[i] = (int)(rand() & 0x5);
	}
	//printArray(h_a, size);

	clock_t cpu_start, cpu_end;
	cpu_start = clock();
	int cpuSum = getSum(h_a, size);
	cpu_end = clock();

	dim3 block(blockSize);
	dim3 grid(size / block.x);
	int rollCount = grid.x / 512;
	printf("Kernel launch parameters | grid.x: %d, block.x: %d\n", 512, block.x);

	int* d_a, * d_part, * pinned_d_a, * pinned_d_part;

	gpuErrorchk(cudaMalloc((void**)&d_a, bSize));
	gpuErrorchk(cudaMalloc((void**)&d_part, partSize));
	gpuErrorchk(cudaMalloc((void**)&pinned_d_a, bSize));
	gpuErrorchk(cudaMalloc((void**)&pinned_d_part, partSize));

	clock_t htod_start, htod_end;
	htod_start = clock();
	gpuErrorchk(cudaMemcpy(d_a, h_a, bSize, cudaMemcpyHostToDevice));
	gpuErrorchk(cudaMemset(d_part, 0, partSize));
	htod_end = clock();

	clock_t gpu_start, gpu_end;
	gpu_start = clock();
	reduction_unrolling_intersum << <512, block >> > (d_a, d_part, size, rollCount);
	gpuErrorchk(cudaDeviceSynchronize());
	gpu_end = clock();

	clock_t dtoh_start, dtoh_end;
	dtoh_start = clock();
	gpuErrorchk(cudaMemcpy(h_part, d_part, partSize, cudaMemcpyDeviceToHost));
	dtoh_end = clock();

	int gpu = 0;
	for (int i = 0;i < (grid.x);i++) {
		gpu += h_part[i];
	}
	compare_arrays(&cpuSum, &gpu, 1);

	clock_t p_htod_start, p_htod_end;
	p_htod_start = clock();
	gpuErrorchk(cudaMemcpy(pinned_d_a, pinned_h_a, bSize, cudaMemcpyHostToDevice));
	gpuErrorchk(cudaMemset(pinned_d_part, 0, partSize));
	p_htod_end = clock();

	//rollCount = 2048;
	clock_t reduced_start, reduced_end;
	reduced_start = clock();
	reduction_unrolling_intersum << <512, block >> > (pinned_d_a, pinned_d_part, size, rollCount);
	gpuErrorchk(cudaDeviceSynchronize());
	reduced_end = clock();

	clock_t p_dtoh_start, p_dtoh_end;
	p_dtoh_start = clock();
	gpuErrorchk(cudaMemcpy(pinned_h_a, pinned_d_part, partSize, cudaMemcpyDeviceToHost));
	p_dtoh_end = clock();

	//printArray(h_part, grid.x);
	gpu = 0;
	for (int i = 0;i < (grid.x);i++) {
		gpu += h_part[i];
	}

	compare_arrays(&cpuSum, &gpu, 1);

	printExecution("CPU", cpu_start, cpu_end);
	printExecution("UNPINNED htod", htod_start, htod_end);
	printExecution("UNPINNED dtoh", dtoh_start, dtoh_end);
	printExecution("UNPINNED", gpu_start, gpu_end);
	printExecution("PINNED htod", p_htod_start, p_htod_end);
	printExecution("PINNED dtoh", p_dtoh_start, p_dtoh_end);
	printExecution("PINNED", reduced_start, reduced_end);
	printExecution("Sum of UNPINNED tasks", gpu_start + dtoh_start + htod_start, gpu_end + dtoh_end + htod_end);
	printExecution("Sum of PINNED tasks", reduced_start + p_dtoh_start + p_htod_start, reduced_end + p_dtoh_end + p_htod_end);


	cudaFree(d_a);
	cudaDeviceReset();
}

void pinnedMemExample(int size) {

	int bSize = size * sizeof(float);

	float* h_a, * pinned_h_a, * d_a, * pinned_d_a;
	h_a = (float*)malloc(bSize);
	gpuErrorchk(cudaMalloc((float**)&d_a, bSize));
	gpuErrorchk(cudaMalloc((float**)&pinned_d_a, bSize));
	gpuErrorchk(cudaMallocHost(&pinned_h_a, bSize));
	gpuErrorchk(cudaMalloc((float**) & pinned_d_a, bSize));

	for (int i = 0; i < size;i++) {
		h_a[i] = pinned_h_a[i] = 7;
	}



	clock_t htod_start, htod_end;
	htod_start = clock();
	gpuErrorchk(cudaMemcpy(d_a, h_a, bSize, cudaMemcpyHostToDevice));
	htod_end = clock();

	clock_t dtoh_start, dtoh_end;
	dtoh_start = clock();
	gpuErrorchk(cudaMemcpy(h_a, d_a, bSize, cudaMemcpyDeviceToHost));
	dtoh_end = clock();

	clock_t p_htod_start, p_htod_end;
	p_htod_start = clock();
	gpuErrorchk(cudaMemcpy(pinned_d_a, pinned_h_a, bSize, cudaMemcpyHostToDevice));
	p_htod_end = clock();

	clock_t p_dtoh_start, p_dtoh_end;
	p_dtoh_start = clock();
	gpuErrorchk(cudaMemcpy(pinned_h_a, pinned_d_a, bSize, cudaMemcpyDeviceToHost));
	p_dtoh_end = clock();

	printExecution("htod", htod_start, htod_end);
	printExecution("dtoh", dtoh_start, dtoh_end);
	printExecution("PINNED htod", p_htod_start, p_htod_end);
	printExecution("PINNED dtoh", p_dtoh_start, p_dtoh_end);

	cudaFree(d_a);
	cudaFree(pinned_d_a);
	cudaDeviceReset();
}

void transposeExample(int nx, int ny) {

	int block_x = 128;
	int block_y = 8;

	int size = nx * ny;
	int bSize = sizeof(int*) * size;

	printf("Transposing Matrix of %dx%d size with block size %dx%d\n", nx, ny, block_x, block_y);

	int* h_mat = (int*)malloc(bSize);
	int* h_trans = (int*)malloc(bSize);
	int* test = (int*)malloc(bSize);

	generateRandArray(h_mat, 0xA, size);

	transpose(h_mat, test, nx, ny);
	//printArray(test, size);

	int* d_mat, *d_trans;
	gpuErrorchk(cudaMalloc((void**)&d_mat, bSize));
	gpuErrorchk(cudaMalloc((void**)&d_trans, bSize));

	gpuErrorchk(cudaMemcpy(d_mat, h_mat, bSize, cudaMemcpyHostToDevice));

	dim3 blocks(block_x, block_y);
	dim3 grid(nx/block_x, ny/block_y);

	clock_t gpu_start, gpu_end;
	gpu_start = clock();
	transpose_row_to_column << <grid, blocks>> > (d_mat, d_trans, nx, ny);
	gpuErrorchk(cudaDeviceSynchronize());
	gpu_end = clock();

	gpuErrorchk(cudaMemcpy(d_trans, h_trans, bSize, cudaMemcpyHostToDevice));

	clock_t coal_start, coal_end;
	coal_start = clock();
	transpose_column_to_row << <grid, blocks >> > (d_mat, d_trans, nx, ny);
	gpuErrorchk(cudaDeviceSynchronize());
	coal_end = clock();

	gpuErrorchk(cudaMemcpy(d_trans, h_trans, bSize, cudaMemcpyHostToDevice));

	int rollCount = 32;
	dim3 rollGrid(grid.x / rollCount, grid.y / rollCount);
	clock_t roll_start, roll_end;
	roll_start = clock();
	transpose_unrolling << <grid, blocks >> > (d_mat, d_trans, nx, ny, rollCount);
	gpuErrorchk(cudaDeviceSynchronize());
	roll_end = clock();

	gpuErrorchk(cudaMemcpy(h_trans, d_trans, bSize, cudaMemcpyDeviceToHost));

	//printArray(h_trans, size);

	compare_arrays(test, h_trans, size);

	printExecution("Strided Writing", gpu_start, gpu_end);
	printExecution("Coalesced Writing", coal_start, coal_end);
	printExecution("Unrolled", roll_start, roll_end);

	cudaDeviceReset();




}

void sharedMemExample(int size) {

	int blockSize = SHARED_ARRAY_SIZE;
	int bSize = sizeof(int) * size;

	int* h_in = (int*)malloc(bSize);
	generateRandArray(h_in, 0xA, size);
	int* h_out = (int*)malloc(bSize);
	int* d_in, * d_out;
	gpuErrorchk(cudaMalloc((int**)&d_in, bSize));
	gpuErrorchk(cudaMalloc((int**)&d_out, bSize));


	dim3 block(SHARED_ARRAY_SIZE);
	dim3 grid(size / block.x + 1);

	gpuErrorchk(cudaMemcpy(d_in, h_in, bSize, cudaMemcpyHostToDevice));

	//Static call
	clock_t static_start, static_end;
	static_start = clock();
	smem_static_test << <grid, block >> > (d_in, d_out, size);
	gpuErrorchk(cudaDeviceSynchronize());
	static_end = clock();

	gpuErrorchk(cudaMemcpy(h_out, d_out, bSize, cudaMemcpyDeviceToHost));

	compare_arrays(h_out, h_in, size);

	gpuErrorchk(cudaMemcpy(d_in, h_in, bSize, cudaMemcpyHostToDevice));

	//Dynamic call
	clock_t dyn_start, dyn_end;
	dyn_start = clock();
	smem_dynamic_test << <grid, block, sizeof(int) * SHARED_ARRAY_SIZE >> > (d_in, d_out, size);
	gpuErrorchk(cudaDeviceSynchronize());
	dyn_end = clock();

	gpuErrorchk(cudaMemcpy(h_out, d_out, bSize, cudaMemcpyDeviceToHost));

	compare_arrays(h_out, h_in, size);

	printExecution("Static", static_start, static_end);
	printExecution("Dynamic", dyn_start, dyn_end);


	cudaFree(d_in);
	cudaFree(d_out);
	cudaDeviceReset();

}

void sharedAccessExample(int memconfig){


	if (memconfig == 1)
	{
		cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
	}
	else
	{
		cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
	}

	
	cudaSharedMemConfig pConfig;
	cudaDeviceGetSharedMemConfig(&pConfig);
	printf("with Bank Mode:%s ", pConfig == 1 ? "4-Byte" : "8-Byte");
	

	// set up array size 2048
	int nx = BDIMX;
	int ny = BDIMY;

	bool iprintf = 0;
	
	iprintf = memconfig;

	size_t nBytes = nx * ny * sizeof(int);

	// execution configuration
	dim3 block(BDIMX, BDIMY);
	dim3 grid(1, 1);
	printf("<<< grid (%d,%d) block (%d,%d)>>>\n", grid.x, grid.y, block.x,
		block.y);

	// allocate device memory
	int *d_C;
	cudaMalloc((int**)&d_C, nBytes);
	int *gpuRef = (int *)malloc(nBytes);

	cudaMemset(d_C, 0, nBytes);
	setColReadRow << <grid, block >> >(d_C);
	cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

	if (iprintf)  printData("set col read col   ", gpuRef, nx * ny);

	cudaMemset(d_C, 0, nBytes);
	setRowReadRow << <grid, block >> >(d_C);
	cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

	if (iprintf)  printData("set row read row   ", gpuRef, nx * ny);

	cudaMemset(d_C, 0, nBytes);
	setRowReadCol << <grid, block >> >(d_C);
	cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

	if (iprintf)  printData("set row read col   ", gpuRef, nx * ny);

	// free host and device memory
	cudaFree(d_C);
	free(gpuRef);

	// reset device
	cudaDeviceReset();
	//return EXIT_SUCCESS;
}

void convolution(unsigned char* h_data, unsigned char* h_newData, double* filter, int rows, int cols, int channels, int size, int kernSize) {
	int bSize = size * sizeof(unsigned char);
	int filSize = sizeof(double) * kernSize * kernSize;
	int grid_x = rows;
	int grid_y = channels;
	int blockSize = 32;
	int grid_z = cols / blockSize;

	//for (int i = 0;i < kernSize;i++) {
	//	for (int j = 0;j < kernSize;j++) {
	//		printf("%d ", filter[i * kernSize + j]);
	//	}
	//	printf("\n");
	//}

	printf("Convoluting Image of %dx%d size with block size %d\n", rows, cols, blockSize);

	//cpu convolution
	//printArray(test, size);

	unsigned char* d_data, * d_newData;
	double* d_filter;
	gpuErrorchk(cudaMalloc((void**)&d_data, bSize));
	gpuErrorchk(cudaMalloc((void**)&d_newData, bSize));
	gpuErrorchk(cudaMalloc((void**)&d_filter, bSize));

	clock_t gpu_start, gpu_end;
	gpu_start = clock();
	gpuErrorchk(cudaMemcpy(d_data, h_data, bSize, cudaMemcpyHostToDevice));
	gpuErrorchk(cudaMemcpy(d_filter, filter, filSize, cudaMemcpyHostToDevice));

	dim3 grid(grid_x, grid_y, grid_z);
	dim3 block(blockSize);

	convolutionKernel << <grid, block >> > (d_data, d_newData, d_filter, size, kernSize);
	gpuErrorchk(cudaDeviceSynchronize());

	gpuErrorchk(cudaMemcpy(h_newData, d_newData, bSize, cudaMemcpyDeviceToHost));
	gpu_end = clock();

	//for (int i = 0;i < 100000;i++) {
	//	printf("pixel %d: %d\n", i, h_newData[i]);
	//}
	//printArray(h_trans, size);

	//compare_arrays(test, h_trans, size);

	printExecution("GPU execution", gpu_start, gpu_end);

	cudaDeviceReset();
	
}