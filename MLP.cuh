#pragma once
#include "MLP.h"




using namespace std;

class Paratron : public Perceptron{
public:
	Paratron(int inputs, activation_function func, double bias = 1.0);
	int inputSize;
	double* d_weights;
	double* d_bufferProduct;
	double* d_vectorInput;
	double bias;
	double run(vector<double> x);
	double run(double* d_x);

	//vector<double> weights;
	//double bias;
	//activation_function A_F;
};

class MultiLayerParatron : public MultiLayerPerceptron{
public:
	MultiLayerParatron(vector<int> cells_in_layer, loss_function func, double bias = 1.0, double eta = 0.5, int batchSize = 0);
	void addLayer(int CIL, activation_function func);
	void finalize();
	vector<double> run(vector<double> x);
	void run(double* d_x);
	vector<double> getRun(double* d_x);
	vector<vector<double>> getOut(double* d_x);
	void cleanRun(double* d_x);
	void cleanerRun(double* d_x);
	vector<double> getCleanRun(double* d_x);
	void batchRun(double* d_batchX);
	vector<vector<double>> getBatchRun(double* d_batchX);
	double getLoss(vector<double> x, vector<double> y);
	void getLoss(double* x, double* y);
	void bLoss(double* x, double* y);
	double bp(vector<double> x, vector<double> y);
	double bp(double* x, vector<double> y);
	double bp(double* x, double* y);
	double bugBp(double** x, double** y);
	double batchBp(double* x, double* y);
	double cleanbp(double* x, vector<double> y);
	double cleanbp(double* x, double* y);
	double cleanerbp(double* x, double* y);
	double batchP(double* batchX, double* batchY);
	vector<vector<Paratron> > network;
	vector<vector<vector<double>>> h_weights;
	vector<vector<int>> weight_lengths;
	int batchSize;
	int termSize;
	int outputSize;
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

	//vector<int> cells_in_layer;
	//double bias;
	//double eta;
	//loss_function L_F;
	//vector<vector<double> > outputs;
	//vector<vector<double> > error_terms;
};













