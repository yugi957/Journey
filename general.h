#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>;
#include <cstring>
#include <time.h>
#include <string>
#include <vector>
#include <iterator>
#include <random>
#include <numeric>
#include <fstream>
#include <type_traits>

using namespace std;

struct vec3 {
	int x, y, z;
	vec3() : x(0), y(0), z(0) {}
	vec3(int x, int y, int z) : x(x), y(y), z(z) {}
};

void xavier_init(vector<vector<float>>& weights, int input_size, int output_size);

void he_init(std::vector<std::vector<float>>& weights, int input_size, int output_size);

void xavier_init_conv(vector<vector<vector<vector<float>>>>& conv_weights, int input_channels, int output_channels, int kernel_size);

void he_init_conv(vector<vector<vector<vector<float>>>>& conv_weights, int input_channels, int output_channels, int kernel_size);

float frand();

float getSum(vector<float> x);

float max(vector<float> x);

char* createFilename(char* path, string name, char* extension);

void generateRandArray(int* arr, int scale, int size);

void printArray(int* arr, int size);
void printArray(float* arr, int size);

void printExecution(char* s, clock_t start, clock_t end);

void sum_array_cpu(int* a, int* b, int* c, int size);

void sum_arrays_cpu(int* a, int* b, int* c, int* d, int size);

void compare_arrays(int* a, int* b, int size);
void compare_arrays(float* a, float* b, int size);


int getSum(int* a, int size);
int getSum(float* a, int size);

void transpose(int* mat, int* trans, int nx, int ny);
void average3D(vector<vector<vector<float>>>* a, vector<vector<float>>* b);
void compare3D(vector<vector<vector<float>>> a, vector<vector<vector<float>>> b);
bool compare2D(vector<vector<float>> a, vector<vector<float>> b);
void compare5D(vector<vector<vector<vector<vector<float>>>>> a, vector<vector<vector<vector<vector<float>>>>> b);
void compareHtoConvWeight(vector<vector<vector<float>>> a, vector<vector<vector<vector<vector<float>>>>> b);

void shuffleData(vector<vector<float>>& images, vector<vector<float>>& labels);
//vector<vector<vector<float>>> batchify(vector<vector<float>>* data, int batchSize);

template <typename T>
vector<T> flatten4D(vector<vector<vector<vector<T>>>>& to_flatten) {
	vector<T> flattened = {};
	for (auto a : to_flatten)
		for (auto b : a)
			for (auto c : b)
				for (auto val : c)
					flattened.push_back(val);
	return flattened;
}

vector<vector<float>> autoencode(vector<vector<float>> set, int size);

void write4D(vector<vector<vector<vector<float>>>> vec4d);
void write2D(vector<vector<float>>);