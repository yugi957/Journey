#include "general.h"

void xavier_init(std::vector<std::vector<double>>& weights, int input_size, int output_size) {
	// Define a random number generator
	std::default_random_engine generator;
	std::normal_distribution<double> distribution(0.0, 1.0);

	// Resize the weights matrix to the appropriate size
	weights.resize(output_size, std::vector<double>(input_size));

	// Compute the scaling factor
	double scaling_factor = std::sqrt(6.0 / (input_size + output_size));

	// Fill the weights matrix with random values
	for (int i = 0; i < output_size; i++) {
		for (int j = 0; j < input_size; j++) {
			weights[i][j] = distribution(generator) * scaling_factor;
		}
	}
}

double frand() {
	return (2.0 * (double)rand() / RAND_MAX) - 1.0;
}

double getSum(vector<double> x) {
	double sum = 0;
	for (int i = 0;i < x.size();i++) {
		sum += x[i];
	}
	return sum;
}

double max(vector<double> x) {
	double max = 0;
	for (int i = 0;i < x.size();i++) {
		if (x[i] > max) max = x[i];
	}
	return max;
}

char* createFilename(char* path, string name, char* extension) {
	string file = (path + name + extension);
	char* filename = new char[file.length() + 1];
	strcpy(filename, file.c_str());
	return filename;
}

void generateRandArray(int* arr, int scale, int size) {
	time_t t;
	srand((unsigned)time(&t));
	for (int i = 0;i < size;i++) {
		arr[i] = (int)(rand() & scale);
	}
}

void printArray(int* arr, int size) {
	printf("[");
	for (int i = 0;i < size;i++) {
		printf("% d, ", arr[i]);
	}
	printf("]\n");
}
void printArray(double* arr, int size) {
	printf("[");
	for (int i = 0;i < size;i++) {
		printf("% f, ", arr[i]);
	}
	printf("]\n");
}

void printExecution(char* s, clock_t start, clock_t end) {
	printf("%s: %4.7f\n", s, (double)((double)(end - start) / CLOCKS_PER_SEC));
}

void sum_array_cpu(int* a, int* b, int* c, int size) {
	for (int i = 0;i < size;i++) {
		c[i] = a[i] + b[i];
	}
}

void sum_arrays_cpu(int* a, int* b, int* c, int* d, int size) {

	for (int i = 0;i < size;i++) {
		d[i] = a[i] + b[i] + c[i];
	}
}

void compare_arrays(int* a, int* b, int size) {
	for (int i = 0;i < size;i++) {
		if (a[i] != b[i]) {
			printf("%d :: %d\n", a[i], b[i]);
			printf("\nARRAYS ARE DIFFERENT\n\n");
			return;
		}
	}
	printf("SUCCESS Arrays are the same\n");
}

void compare_arrays(double* a, double* b, int size) {
	for (int i = 0;i < size;i++) {
		if (a[i] != b[i]) {
			printf("%d :: %d\n", a[i], b[i]);
			printf("\nARRAYS ARE DIFFERENT\n\n");
			return;
		}
	}
	printf("SUCCESS Arrays are the same\n");
}

int getSum(int* a, int size) {
	int sum = 0;
	for (int i = 0;i < size;i++) {
		sum += a[i];
	}
	return sum;
}

int getSum(double* a, int size) {
	double sum = 0;
	for (int i = 0;i < size;i++) {
		sum += a[i];
	}
	return sum;
}

void transpose(int* mat, int* trans, int nx, int ny) {
	for (int iy = 0;iy < ny;iy++) {
		for (int ix = 0;ix < nx;ix++) {
			trans[ix * ny + iy] = mat[iy * nx + ix];
		}
	}
}