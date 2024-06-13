
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

void cudaAllocate2dOffVectorHostRef(float*** d_inputs, vector<vector<float>> h_inputs);
void cudaFree2dHostRef(float*** d_a, int size);
void cudaMemCopy2dOffVectorHostRef(float*** d_a, vector<vector<float>> h_a);

void cudaAllocateFull2dOffVectorHostRef(float*** d_a, vector<vector<float>> h_a, int batchSize);

void cudaAllocate3dOffVectorHostRef(float*** d_a, vector<vector<vector<float>>> h_a);

void cudaAllocate2dOffVector(float** d_a, vector<vector<float>> h_inputs, int** lengths);
void cudaAllocate2dOffVector(float** d_a, vector<vector<float>> h_inputs);

void cudaAllocate3dOffVector(float** d_inputs, vector<vector<vector<float>>> h_inputs);

void cudaMemcpy2dOffVector(float** d_a, vector<vector<float>> h_inputs);

void cudaMemcpy3dOffVector(float** d_a, vector<vector<vector<float>>> h_inputs);

void cudaMemcpy3dOffVectorHostRef(float*** d_a, vector<vector<vector<float>>> h_a);

vector<vector<float>> cudaCopy2dBackToVector(float** d_a, vector<int> lengths);

vector<vector<vector<float>>> cudaCopy3dBackToVector(float** d_a, vector<vector<int>> lengths);
vector<vector<float>> cudaCopy2dBackToVectorHref(float** d_a, vector<int> lengths);
vector<vector<vector<float>>> cudaCopy3dBackToVectorHref(float*** d_a, vector<vector<int>> lengths);
vector<vector<float>> cudaCopyBatchBackToVectorHref(float** d_a, int size, int batchSize);

vector<float> cudaCopy2dBackTo1dVector(float** d_a, vector<int> lengths);

float*** createBatches(float** hr_a, int batchSize, int examples, int size);
vector<vector<float>> batchify(vector<vector<float>>* data, int batchSize);