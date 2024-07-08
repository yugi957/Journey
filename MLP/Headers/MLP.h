#pragma once
#include "../../general.h"
#include "../../IP/Headers/ImageProcessing.h"
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
	SOFTMAX,
	LEAKY_RELU,
	PLACEHOLDER
};

enum loss_function {
	MSE,
	CROSS_ENTROPY
};

enum layer_type {
	INPUT,
	DENSE,
	CONV,
	MAX_POOL,
	OUTPUT
};

class MultiLayerPerceptron {
public:
	MultiLayerPerceptron();
	MultiLayerPerceptron(vector<int> cells_in_layer, loss_function func, float bias = 1.0, float eta = 0.01, float momentum = .4, int batchSize = 0);
	void initializeWeights();
	void addLayer(int CIL, activation_function func, vec3 dims = vec3(-1, 1, 1));
	void addConv(size_t kernel_size, size_t padding, size_t stride, size_t out_depth, activation_function func);
	void addMaxPool(size_t kernel_size, size_t padding, size_t stride);
	void finalize();
	float run(vector<float>& x, vector<float>& w, activation_function A_F, int layer);
	vector<float> softmax(vector<float> x, vector<vector<float>> w);
	float activation(float x, activation_function A_F);
	vector<float> Wrun(vector<float>& x);
	void activate_conv(vector<float>& x, activation_function A_F);
	vector<float> forward_conv(vector<float> x);
	vector<vector<float>> batchRun(vector<vector<float>> x);
	vector<vector<float>> Wout(vector<float> x);
	float getLoss(vector<float>& x, vector<float>& y);
	float Wbp(vector<float> x, vector<float> y);
	float Mbp(vector<float> x, vector<float> y);
	float backward_conv(vector<float> x, vector<float> y);
	void train(vector<vector<float>> train_set, vector<vector<float>> label_set, int epochs, int progressCheck);

	vector<layer_type> layers;
	vector<int> cells_in_layer;
	vector<vec3> dimensions;
	float bias;
	float eta;
	float momentum;
	int batchSize;
	loss_function L_F;
	vector<activation_function> A_Fs;
	vector<vector<vector<float>>> h_weights;
	vector<vector<vector<vector<vector<float>>>>> conv_weights;
	vector<vector<size_t>> conv_params;
	vector<vector<float> > outputs;
	vector<vector<vector<float>>> batch_outputs;
	vector<vector<float> > error_terms;
	vector<vector<vector<float>>> batch_ETs;
	vector<vector<vector<vector<float>>>> batch_gradients;
	vector<vector<vector<float>>> gradient;
	vector<vector<vector<vector<vector<float>>>>> conv_gradient;
	vector<vector<vector<float>>> moments;

};