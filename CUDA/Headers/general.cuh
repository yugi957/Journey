
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>;
#include <vector>
#include "../../general.h"

using namespace std;
#define gpuErrorchk(ans) {	gpuAssert((ans), __FILE__, __LINE__);}

inline void gpuAssert(cudaError code, const char* file, int line);

void cudaAllocate2dOffVectorHostRef(double*** d_inputs, vector<vector<double>> h_inputs);
void cudaFree2dHostRef(double*** d_a, int size);
void cudaMemCopy2dOffVectorHostRef(double*** d_a, vector<vector<double>> h_a);

void cudaAllocateFull2dOffVectorHostRef(double*** d_a, vector<vector<double>> h_a, int batchSize);

void cudaAllocate3dOffVectorHostRef(double*** d_a, vector<vector<vector<double>>> h_a);

void cudaAllocate2dOffVector(double** d_a, vector<vector<double>> h_inputs, int** lengths);
void cudaAllocate2dOffVector(double** d_a, vector<vector<double>> h_inputs);

void cudaAllocate3dOffVector(double** d_inputs, vector<vector<vector<double>>> h_inputs);

void cudaMemcpy2dOffVector(double** d_a, vector<vector<double>> h_inputs);

void cudaMemcpy3dOffVector(double** d_a, vector<vector<vector<double>>> h_inputs);

void cudaMemcpy3dOffVectorHostRef(double*** d_a, vector<vector<vector<double>>> h_a);

vector<vector<double>> cudaCopy2dBackToVector(double** d_a, vector<int> lengths);

vector<vector<vector<double>>> cudaCopy3dBackToVector(double** d_a, vector<vector<int>> lengths);
vector<vector<double>> cudaCopy2dBackToVectorHref(double** d_a, vector<int> lengths);
vector<vector<vector<double>>> cudaCopy3dBackToVectorHref(double*** d_a, vector<vector<int>> lengths);
vector<vector<double>> cudaCopyBatchBackToVectorHref(double** d_a, int size, int batchSize);

vector<double> cudaCopy2dBackTo1dVector(double** d_a, vector<int> lengths);

double*** createBatches(double** hr_a, int batchSize, int examples, int size);
vector<vector<double>> batchify(vector<vector<double>>* data, int batchSize);