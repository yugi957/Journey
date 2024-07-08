#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <time.h>
#include <math.h>
#include "../../MLP/Headers/MLP.cuh"
#include "../../MLP/Headers/MLP.h"


__global__ void copyElements(float* dest, float* src, int size) {
	int gid = blockDim.x * blockIdx.x + threadIdx.x;

	if (gid >= size) return;
	dest[gid] = src[gid];
}

__global__ void copySeqElements(float* dest, float* src, int size) {
	for (int i = 0;i < size;i++) {
		dest[i] = src[i];
	}
}

__global__ void SoftMaxSeq(float* output, int size) {
	float denom = 0;
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

__global__ void runCleanParatron(float* input, float* output, float* weights, activation_function A_F, int inputSize, int CIL, float bias) {

	//inputs will be passed from parent kernel as output + 0
	//output array will be passed as output + CIL[layer]

	//Maybe block will be layer and thread will be cell
	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	if (gid > CIL) return;
	int cellOffset = (inputSize + 1) * gid;
	float* localWeights = weights + cellOffset;

	float sum = 0;
	for (int i = 0;i < inputSize;i++) {
		sum += input[i] * localWeights[i];
	}
	sum += bias * localWeights[inputSize];
	switch (A_F) {
	case SIGMOID:
		sum = 1 / (1 + exp(-sum)); //sigmoid
		break;
	case RELU:
		sum = (sum > 0) ? sum : (.1 * sum); //ReLu
		break;
	case SOFTMAX:
		sum = exp(sum);
		break;
	}
	output[gid] = sum;
}

__global__ void getLossSeq(float* x, float* y, float* loss, loss_function* L_F, int size) {
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
			//printf("gpu loss[%d]: %f\n", i, *loss);
			//printf("prediction: %f ; actual: %f\n", x[i], y[i]);
			//printf("clean: %f * %f = %f\n", y[i], log(x[i]), y[i] * log(x[i]));
		}
		break;
	}
	//printf("\n");
	//printf("clean loss: %f\n", *loss);
}

__global__ void cleanGradient(float* weights, float* for_terms, float* terms, float* outputs, int for_CIL, int CIL, activation_function A_F) {
	int cid = blockIdx.x * blockDim.x + threadIdx.x; //cell Id given error_term layer ; weights point to this cell
	if (cid >= CIL) return;

	float err_sum = 0.0;

	for (int i = 0;i < for_CIL;i++) {
		int cellOffset = (CIL + 1) * i;
		err_sum += weights[cellOffset + cid] * for_terms[i];
	}
	float derivative;
	switch (A_F) {
	case(SIGMOID):
		derivative = outputs[cid] * (1 - outputs[cid]);
		break;
	case(RELU):
		derivative = (outputs[cid] > 0) ? 1.0f : 0.1f;
		break;
	}
	terms[cid] = derivative * err_sum;
	if (cid == CIL) {
		printf("HELLO\n");
	}
}

__global__ void cleanUpdateWeightsbyLayer(float* weights, float* error_terms, float* outputs, float eta, int CIL, int forCIL, float bias) {
	if (threadIdx.x > forCIL || blockIdx.x > CIL) return;
	int wid = gridDim.x * threadIdx.x + blockIdx.x;
	float delta = eta * error_terms[threadIdx.x] * outputs[blockIdx.x];
	//if (threadIdx.x == 0 && blockIdx.x == 0) printf("weight: %f\n", weights[wid]);
	if (blockIdx.x == CIL) delta = eta * error_terms[threadIdx.x] * bias;
	weights[wid] -= delta;
}

__global__ void batchCopy(float* dest, float* src, int size) {
	int gid = blockDim.x * blockIdx.x + threadIdx.x;

	if (gid > size) return;
	//src[gid] = 5;
	dest[gid] = src[gid];
}

__global__ void batchSoftMax(float* output, int size, int batchSize) {
	if (threadIdx.x > batchSize) return;
	//printf("thread:%d\n", threadIdx.x);
	int batch = threadIdx.x;
	float* local_output = output + batch * size;
	float denom = 0;
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

__global__ void runBatchParatron(float* input, float* output, float* weights, activation_function A_F, int inputSize, int CIB, int CIL, float bias) {
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
	float* localWeights = weights + cellOffset;

	int inputOffset = blockIdx.x * inputSize;
	float* localInput = input + inputOffset;
	bool check = false;

	float sum = 0;
	for (int i = 0;i < inputSize;i++) {
		sum += localInput[i] * localWeights[i];
	}
	sum += bias * localWeights[inputSize];
	switch (A_F) {
	case SIGMOID:
		sum = 1 / (1 + exp(-sum)); //sigmoid
		break;
	case RELU:
		sum = (sum > 0) ? sum : 0; //ReLu
		break;
	case LEAKY_RELU:
		sum = (sum > 0) ? sum : (.1 * sum); //ReLu
		break;
	case SOFTMAX:
		sum = exp(sum);
		break;
	}
	output[gid] = sum;
}

__global__ void batchLoss(float* x, float* y, float* loss, loss_function L_F, int size, int batchSize) {
	int batch = threadIdx.x;

	float sum = 0.0;
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
			float testx = 0.0;
			float testy = 0.0;
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

__global__ void getErrorLayerWRTInputSeq(float* error_terms, float* x, float* y, int size, loss_function L_F, activation_function A_F) {
	for (int i = 0;i < size;i++) {
		//error_terms[i] = x[i] * (1 - y[i]) * 2 * (x[i] - y[i]);
		error_terms[i] = x[i] - y[i];
		//printf("error_term[%d]: %f\n", i, error_terms[i]);
		//printf("output[%d]: %f\n", i, x[i]);
		//switch (L_F) {
		//case(MSE):
		//	float activation;
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
	//printf("\n");
}

__global__ void batchErrorLayer(float* error_terms, float* x, float* y, int size, loss_function L_F, activation_function A_F) {
	int batch = threadIdx.x;
	float* local_x = x + (batch * size);
	float* local_y = y + (batch * size);
	for (int i = 0;i < size;i++) {
		switch (L_F) {
		case(MSE):
			error_terms[i] = 2 * (local_x[i] - local_y[i]);
			break;
		case(CROSS_ENTROPY):
			error_terms[i] = local_x[i] - local_y[i];
			break;
		}
	}
}

__global__ void batchGradient(float* weights, float* for_terms, float* terms, float* outputs, int for_CIL, int CIL, int CIB, activation_function A_F) {

	//gridDim.x refers to batchSize, so blockIdx.x is which row (batch) we are in, and blockIdx.y is which multiple of 32 of the cell we are in
	int cid = blockIdx.y * blockDim.x + threadIdx.x; //cell Id given error_term layer ; weights point to this cell
	int gid = (CIL * blockIdx.x) + (blockIdx.y * blockDim.x) + threadIdx.x;
	if (cid >= CIL) return;
	if (gid >= CIB) return;

	int for_offset = blockIdx.x * for_CIL;
	//if (for_CIL == 10 && blockIdx.x == 0 && blockIdx.y == for_CIL / 32 && threadIdx.x == 31) printf("gridDim.x: %d, gridDim.y: %d\n", gridDim.x, gridDim.y);
	//if (for_CIL == 10 && blockIdx.x == gridDim.x - 1 && blockIdx.y == gridDim.y-1 && threadIdx.x == 0) printf("batch: %d, block: %d, thread: %d\n", blockIdx.x, blockIdx.y, threadIdx.x);

	int offset = blockIdx.x * CIL;
	float err_sum = 0.0;

	for (int i = 0;i < for_CIL;i++) {
		int cellOffset = (CIL + 1) * i;
		err_sum += weights[cellOffset + cid] * for_terms[i + for_offset];
	}
	float derivative;
	switch (A_F) {
	case(SIGMOID):
		derivative = outputs[cid + offset] * (1 - outputs[cid + offset]);
		break;
	case(RELU):
		derivative = (outputs[cid + offset] > 0) ? 1.0f : 0.0f;
		break;
	case(LEAKY_RELU):
		derivative = (outputs[cid + offset] > 0) ? 1.0f : 0.1f;
		break;
	}
	terms[cid + offset] = derivative * err_sum;
}

__global__ void batchMakeGradient(float* grad, float* error_terms, float* outputs, int CIL, int forCIL, float bias, int batchSize) {
	int blockId = blockIdx.x * (CIL + 1) * forCIL;
	int gid = blockId + (CIL + 1) * threadIdx.x + blockIdx.y;
	int wid = (CIL + 1) * threadIdx.x + blockIdx.y;
	if (gid > (CIL + 1) * forCIL * batchSize) return;

	int err_offset = forCIL * blockIdx.x;
	float* local_errors = error_terms + err_offset;

	int out_offset = CIL * blockIdx.x;
	float* local_outs = outputs + out_offset;


	//if (threadIdx.x == 0 && blockIdx.x == 0) printf("weight: %f ; batch: %d ; err_offset: %d ; out_offset: %d\n", weights[wid], batch, err_offset, out_offset);
	float dW = local_errors[threadIdx.x] * local_outs[blockIdx.y];
	if (blockIdx.y == CIL) {
		dW = local_errors[threadIdx.x] * bias;
	}
	grad[gid] = dW;
}



//This memory is really fragmented, but you get way more threads going off of cell than batchSize
__global__ void averageGrad(float* batch_grad, float* gradient, int batchSize, float momentum, float* moments, int size, int for_size) {
	int gid = threadIdx.x * size + blockIdx.x; //cell Id given error_term layer ; weights point to this cell
	int batch = size * for_size;
	if (gid > size * for_size) return;


	float sum = 0.0;
	for (int i = 0;i < batchSize;i++) {
		sum += batch_grad[gid + (i * batch)];
	}
	sum /= batchSize;
	moments[gid] = momentum * moments[gid] + sum;
	gradient[gid] = moments[gid];
	//gradient[gid] = sum;
}

__global__ void avgOut(float* batch_outs, float* outs, int batchSize, int CIL) {
	int cid = blockIdx.x * blockDim.x + threadIdx.x; //cell Id given error_term layer ; weights point to this cell
	if (cid > CIL) return;

	float sum = 0.0;
	for (int i = 0;i < batchSize;i++) {
		sum += batch_outs[cid + (i * CIL)];
	}
	sum /= batchSize;
	//if (CIL == 10) printf("out %d: %f\n", cid, sum);
	outs[cid] = sum;
}

//This memory is really fragmented, but you get way more threads going off of cell than batchSize
__global__ void sumGrad(float* batch_grad, float* gradient, int batchSize, int size, int for_size) {
	int gid = threadIdx.x * size + blockIdx.x; //cell Id given error_term layer ; weights point to this cell
	int batch = size * for_size;
	if (gid > size * for_size) return;


	float sum = 0.0;
	for (int i = 0;i < batchSize;i++) {
		sum += batch_grad[gid + (i * batch)];
	}
	gradient[gid] = sum;
}

__global__ void applyGrad(float* weights, float* gradient, float eta, int size) {
	int gid = threadIdx.x * gridDim.x + blockIdx.x;
	if (gid > size) return;

	weights[gid] -= eta * gradient[gid];
}