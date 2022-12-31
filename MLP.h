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


class Perceptron {
public:
	Perceptron(int inputs, double bias = 1.0);
	vector<double> weights;
	double bias;
	double run(vector<double> x);
	void set_weights(vector<double> w_init);
	double activation(double x);
};


/*
class Perceptron {
	public:
		vector<double> weights;
		double bias;
		Perceptron(int inputs, double bias=1.0);
		double run(vector<double> x);
		void set_weights(vector<double> w_init);
		double sigmoid(double x);
};
/*

/*
class MultiLayerPerceptron {
	public:
		MultiLayerPerceptron(vector<int> cells_in_layer, double bias=1.0, double eta=0.5);

		vector<int> cells_in_layer; //num of cells in layer i
		double bias;
		double eta; //learning rate
		vector<vector<Perceptron>> network;
		vector<vector<double>> outputs; // cell outputs
		vector<vector<double>> error_terms; //errors terms of neurons

		void set_weights(vector<vector<vector<double>>> w_init);
		void print_weights();
		vector<double> run(vector<double> x); //I believe x is input
		double bp(vector<double> x, vector<double> y); //must be backprop
};
*/

class MultiLayerPerceptron {
public:
	MultiLayerPerceptron(vector<int> cells_in_layer, double bias = 1.0, double eta = 0.5);
	void set_weights(vector<vector<vector<double> > > w_init);
	void print_weights();
	vector<double> run(vector<double> x);
	double bp(vector<double> x, vector<double> y);

	vector<int> cells_in_layer;
	double bias;
	double eta;
	vector<vector<Perceptron> > network;
	vector<vector<double> > outputs;
	vector<vector<double> > error_terms;
};













