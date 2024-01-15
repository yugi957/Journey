#pragma once
#include "../../general.h"
#include <algorithm>
#include <vector>
#include <iostream>
#include <random>
#include <numeric>
#include <cmath>
#include <time.h>

using namespace std;

enum activation_function {
	SIGMOID,
	RELU,
	SOFTMAX
};

enum loss_function {
	MSE,
	CROSS_ENTROPY
};

class MultiLayerPerceptron {
public:
	MultiLayerPerceptron();
	MultiLayerPerceptron(vector<int> cells_in_layer, loss_function func, double bias = 1.0, double eta = 0.01, int batchSize = 0, double momentum = .4);
	void addLayer(int CIL, activation_function func);
	void finalize();
	double run(vector<double> x, vector<double> w, activation_function A_F, int layer);
	vector<double> softmax(vector<double> x, vector<vector<double>> w);
	double activation(double x, activation_function A_F);
	vector<double> Wrun(vector<double> x);
	vector<vector<double>> batchRun(vector<vector<double>> x);
	vector<vector<double>> Wout(vector<double> x);
	double getLoss(vector<double> x, vector<double> y);
	double Wbp(vector<double> x, vector<double> y);
	double Mbp(vector<double> x, vector<double> y);
	void train(vector<vector<double>> train_set, vector<vector<double>> label_set, int epochs, int progressCheck);

	vector<int> cells_in_layer;
	double bias;
	double eta;
	double momentum;
	int batchSize;
	loss_function L_F;
	vector<activation_function> A_Fs;
	vector<vector<vector<double>>> h_weights;
	vector<vector<double> > outputs;
	vector<vector<vector<double>>> batch_outputs;
	vector<vector<double> > error_terms;
	vector<vector<vector<double>>> batch_ETs;
	vector<vector<vector<vector<double>>>> batch_gradients;
	vector<vector<vector<double>>> gradient;

};