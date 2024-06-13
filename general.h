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

using namespace std;

void xavier_init(vector<vector<float>>& weights, int input_size, int output_size);

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

void shuffleData(vector<vector<float>>& images, vector<vector<float>>& labels);
//vector<vector<vector<float>>> batchify(vector<vector<float>>* data, int batchSize);

vector<vector<float>> autoencode(vector<vector<float>> set, int size);

void write4D(vector<vector<vector<vector<float>>>> vec4d);
void write2D(vector<vector<float>>);