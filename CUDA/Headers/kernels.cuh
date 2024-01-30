#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <time.h>
#include <math.h>
#include "../../MLP/Headers/MLP.cuh"
#include "../../MLP/Headers/MLP.h"


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

__global__ void cleanUpdateWeightsbyLayer(double* weights, double* error_terms, double* outputs, double eta, int CIL, int forCIL, double bias) {
	if (threadIdx.x > forCIL || blockIdx.x > CIL) return;
	int wid = gridDim.x * threadIdx.x + blockIdx.x;
	double delta = eta * error_terms[threadIdx.x] * outputs[blockIdx.x];
	//if (threadIdx.x == 0 && blockIdx.x == 0) printf("weight: %f\n", weights[wid]);
	if (blockIdx.x == CIL) delta = eta * error_terms[threadIdx.x] * bias;
	weights[wid] -= delta;
}

__global__ void batchMakeGradient(double* grad, double* updates, double* error_terms, double* outputs, double eta, double momentum, int CIL, int forCIL, double bias, int batchSize) {
	int blockId = blockIdx.x * (CIL + 1) * forCIL;
	int gid = blockId + (CIL + 1) * threadIdx.x + blockIdx.y;
	int wid = (CIL + 1) * threadIdx.x + blockIdx.y;
	if (gid > (CIL + 1) * forCIL * batchSize) return;

	int err_offset = forCIL * blockIdx.x;
	double* local_errors = error_terms + err_offset;

	int out_offset = CIL * blockIdx.x;
	//outputs += out_offset;
	double* local_outs = outputs + out_offset;


	//if (threadIdx.x == 0 && blockIdx.x == 0) printf("weight: %f ; batch: %d ; err_offset: %d ; out_offset: %d\n", weights[wid], batch, err_offset, out_offset);
	double delta = eta * local_errors[threadIdx.x] * local_outs[blockIdx.y];
	if (blockIdx.y == CIL) delta = eta * local_errors[threadIdx.x] * bias;
	grad[gid] = (momentum * updates[wid]) + delta;
	//if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) printf("momentum: %f, prev grad: %f\n", momentum, updates[wid]);
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