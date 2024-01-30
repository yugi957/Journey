#pragma once
#include "MLP.h"
#include "../../CUDA/Headers/general.cuh"
//#include "../../CUDA/Headers/kernels.cuh"

using namespace std;

class MultiLayerParatron : public MultiLayerPerceptron {
public:
	MultiLayerParatron(vector<int> cells_in_layer, loss_function func, double bias = 1.0, double eta = 0.5, double momentum = .4, int batchSize = 0);
	void addLayer(int CIL, activation_function func);
	void finalize();

	void cleanerRun(double* d_x);
	vector<double> getRun(double* d_x);
	void batchRun(double* d_batchX);
	vector<vector<double>> getBatchRun(double* d_batchX);

	void getLoss(double* x, double* y);
	void bLoss(double* x, double* y);

	double cleanerbp(double* x, double* y);
	vector<vector<double>> getCleanerBp(double* x, double* y);

	double batchP(double* batchX, double* batchY);
	vector<vector<double>> getBatchP(double* batchX, double* batchY);
	double aveBatchP(double* batchX, double* batchY);
	vector<vector<double>> getAveP(double* batchX, double* batchY);

	int* d_CIL;
	double* d_outputs;
	double** d_outputs_href;
	double** d_batch_outs_href;
	int* outputLayerOffsets;
	int* d_outputLayerOffsets;
	double* d_weights;
	double** d_weights_href;
	double** d_gradient_href;
	double** d_batch_grad_href;
	int* d_weightLayerOffsets;
	int* weightLayerOffsets;
	double* d_error_terms;
	double** d_error_terms_href;
	double** d_batch_errors_href;
	activation_function* d_A_Fs;
	loss_function* d_L_F; //I think pointer to pass to GPU
	double* d_loss;
	double* d_eta;

	vector<vector<double>> batch_out;
	vector<vector<double>> batch_err;
	vector<vector<double>> batch_gradient;

	vector<vector<vector<double>>> h_weights;
	vector<vector<int>> weight_lengths;
	int batchSize;
	int termSize;
	int outputSize;
	vector<activation_function> h_A_Fs;
	//vector<int> cells_in_layer;
	//double bias;
	//double eta;
	//loss_function L_F;
	//vector<vector<double> > outputs;
	//vector<vector<double> > error_terms;
};