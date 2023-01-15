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
	MultiLayerParatron(vector<int> cells_in_layer, loss_function func, double bias = 1.0, double eta = 0.5);
	void addLayer(int CIL, activation_function func);
	void finalize();
	vector<double> run(vector<double> x);
	void run(double* d_x);
	vector<double> getRun(double* d_x);
	vector<vector<double>> getOut(double* d_x);
	void cleanRun(double* d_x);
	vector<double> getCleanRun(double* d_x);
	double getLoss(vector<double> x, vector<double> y);
	void getLoss(double* x, double* y);
	double bp(vector<double> x, vector<double> y);
	double bp(double* x, vector<double> y);
	double bp(double* x, double* y);
	double cleanbp(double* x, vector<double> y);
	double cleanbp(double* x, double* y);
	vector<vector<Paratron> > network;
	vector<vector<vector<double>>> h_weights;
	vector<vector<int>> weight_lengths;
	int* d_CIL;
	double* d_outputs;
	double** d_outputs_href;
	int* d_outputLayerOffsets;
	double* d_weights;
	double** d_weights_href;
	int* d_weightLayerOffsets;
	double* d_error_terms;
	double** d_error_terms_href;
	activation_function* d_A_Fs;
	loss_function* d_L_F; //I think pointer to pass to GPU
	double* d_loss;
	double* d_eta;

	//vector<int> cells_in_layer;
	//double bias;
	//double eta;
	//loss_function L_F;
	//vector<vector<double> > outputs;
	//vector<vector<double> > error_terms;
};













