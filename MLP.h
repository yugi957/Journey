#pragma once
#include "general.h"
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

class Perceptron {
public:
	Perceptron();
	Perceptron(int inputs, activation_function func, double bias = 1.0);
	vector<double> weights;
	double bias;
	activation_function A_F;
	double run(vector<double> x);
	double run(vector<double> x, vector<double> w);
	void set_weights(vector<double> w_init);
	double activation(double x);
};

class MultiLayerPerceptron {
public:
	MultiLayerPerceptron();
	MultiLayerPerceptron(vector<int> cells_in_layer, loss_function func, double bias = 1.0, double eta = 0.5);
	void set_weights(vector<vector<vector<double> > > w_init);
	void print_weights();
	void addLayer(int CIL, activation_function func);
	double run(vector<double> x, vector<double> w, activation_function A_F, int layer);
	vector<double> softmax(vector<double> x, vector<vector<double>> w);
	double activation(double x, activation_function A_F);
	double activationDerivative(double x, activation_function A_F);
	vector<double> run(vector<double> x);
	vector<double> Wrun(vector<double> x);
	vector<vector<double>> Wout(vector<double> x);
	double getLoss(vector<double> x, vector<double> y);
	double lossDerivative(double x, double y, loss_function L_F);
	double bp(vector<double> x, vector<double> y);
	double Wbp(vector<double> x, vector<double> y);

	vector<int> cells_in_layer;
	double bias;
	double eta;
	loss_function L_F;
	vector<activation_function> A_Fs;
	vector<vector<Perceptron> > network;
	vector<vector<vector<double>>> h_weights;
	vector<vector<double> > outputs;
	vector<vector<double> > error_terms;
};













