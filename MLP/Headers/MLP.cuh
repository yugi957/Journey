#pragma once
#include "MLP.h"
#include "../../CUDA/Headers/general.cuh"
//#include "../../CUDA/Headers/kernels.cuh"

using namespace std;

class MultiLayerParatron : public MultiLayerPerceptron {
public:
	MultiLayerParatron(vector<int> cells_in_layer, loss_function func, float bias = 1.0, float eta = 0.5, float momentum = .4, int batchSize = 0);
	void addLayer(int CIL, activation_function func);
	void finalize();

	void cleanerRun(float* d_x);
	vector<float> getRun(float* d_x);
	void batchRun(float* d_batchX);
	vector<vector<float>> getBatchRun(float* d_batchX);

	void getLoss(float* x, float* y);
	void bLoss(float* x, float* y);

	float cleanerbp(float* x, float* y);
	vector<vector<float>> getCleanerBp(float* x, float* y);

	float aveBatchP(float* batchX, float* batchY);
	vector<vector<float>> getAveP(float* batchX, float* batchY);

	void toCPU();

	float* d_outputs;
	float** d_outputs_href;
	float** d_batch_outs_href;
	float* d_weights;
	float** d_weights_href;
	float** d_gradient_href;
	float** d_moments_href;
	float** d_batch_moments_href;
	float** d_batch_grad_href;
	float* d_error_terms;
	float** d_error_terms_href;
	float** d_batch_errors_href;
	activation_function* d_A_Fs;
	loss_function* d_L_F; //I think pointer to pass to GPU
	float* d_loss;
	float* d_eta;

	vector<vector<float>> batch_out;
	vector<vector<float>> batch_err;
	vector<vector<float>> batch_gradient;
	cudaStream_t* streams;
	vector<vector<float>> batch_moments;

	vector<vector<vector<float>>> h_weights;
	vector<vector<int>> weight_lengths;
	int batchSize;
	int termSize;
	int outputSize;
	vector<activation_function> h_A_Fs;
	//vector<int> cells_in_layer;
	//float bias;
	//float eta;
	//loss_function L_F;
	//vector<vector<float> > outputs;
	//vector<vector<float> > error_terms;
};