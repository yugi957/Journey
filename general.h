#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>;
#include <cstring>
#include <time.h>
#include <string>
using namespace std;

char* createFilename(char* path, string name, char* extension);

void generateRandArray(int* arr, int scale, int size);

void printArray(int* arr, int size);

void printExecution(char* s, clock_t start, clock_t end);

void sum_array_cpu(int* a, int* b, int* c, int size);

void sum_arrays_cpu(int* a, int* b, int* c, int* d, int size);

void compare_arrays(int* a, int* b, int size);

int getSum(int* a, int size);

void transpose(int* mat, int* trans, int nx, int ny);