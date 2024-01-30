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

void xavier_init(vector<vector<double>>& weights, int input_size, int output_size);

double frand();

double getSum(vector<double> x);

double max(vector<double> x);

char* createFilename(char* path, string name, char* extension);

void generateRandArray(int* arr, int scale, int size);

void printArray(int* arr, int size);
void printArray(double* arr, int size);

void printExecution(char* s, clock_t start, clock_t end);

void sum_array_cpu(int* a, int* b, int* c, int size);

void sum_arrays_cpu(int* a, int* b, int* c, int* d, int size);

void compare_arrays(int* a, int* b, int size);
void compare_arrays(double* a, double* b, int size);


int getSum(int* a, int size);
int getSum(double* a, int size);

void transpose(int* mat, int* trans, int nx, int ny);
void average3D(vector<vector<vector<double>>>* a, vector<vector<double>>* b);
void compare3D(vector<vector<vector<double>>> a, vector<vector<vector<double>>> b);
bool compare2D(vector<vector<double>> a, vector<vector<double>> b);

void shuffleData(vector<vector<double>>& images, vector<vector<double>>& labels);
//vector<vector<vector<double>>> batchify(vector<vector<double>>* data, int batchSize);

vector<vector<double>> autoencode(vector<vector<double>> set, int size);

void write4D(vector<vector<vector<vector<double>>>> vec4d);
void write2D(vector<vector<double>>);