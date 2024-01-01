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

void average3D(vector<vector<vector<double>>>* a, vector<vector<double>>* b) {
	double sum;
	int bSize = a->size();

	for (int l = 0;l < (*a)[0].size();l++) {
		for (int n = 0;n < (*a)[0][l].size();n++) {
			sum = 0;
			for (int b = 0;b < bSize;b++) {
				sum += (*a)[b][l][n];
			}
			(*b)[l][n] = sum / bSize;
		}
	}

}

void compare3D(vector<vector<vector<double>>> a, vector<vector<vector<double>>> b) {
	for (int i = 0;i < a.size();i++) {
		for (int j = 0;j < a[i].size();j++) {
			for (int k = 0;k < a[i][j].size();k++) {
				if (a[i][j][k] != b[i][j][k]) {
					printf("%f :: %f\n", a[i][j][k], b[i][j][k]);
					printf("\nARRAYS ARE DIFFERENT\n\n");
					return;
				}
			}
		}
	}
	printf("SUCCESS Arrays are the same\n");
}

bool compare2D(vector<vector<double>> a, vector<vector<double>> b) {
	for (int i = 0;i < a.size();i++) {
		for (int j = 0;j < a[i].size();j++) {
			if (a[i][j] - b[i][j] < -1 * .000001 && a[i][j] - b[i][j] > .000001) {
				//if(a[i][j] == b[i][j]){
				printf("%f :: %f\n", a[i][j], b[i][j]);
				printf("---- %f ----\n", a[i][j] - b[i][j]);
				printf("DIFFERENCE IN Layer %d, index %d\n", i, j);
				printf("\nARRAYS ARE DIFFERENT\n\n");
				return false;
			}
		}
	}
	printf("SUCCESS Arrays are the same\n");
	return true;
}

void shuffleData(std::vector<std::vector<double>>& images, std::vector<std::vector<double>>& labels) {
	// Seed with a real random value, if available
	std::random_device rd;

	// Create a random number generator
	std::mt19937 g(rd());

	// Create a vector of indices
	std::vector<size_t> indices(images.size());
	std::iota(indices.begin(), indices.end(), 0);

	// Shuffle the indices
	std::shuffle(indices.begin(), indices.end(), g);

	// Create temporary vectors to hold the shuffled data
	std::vector<std::vector<double>> shuffledImages(images.size());
	std::vector<std::vector<double>> shuffledLabels(labels.size());

	// Rearrange the data according to the shuffled indices
	for (size_t i = 0; i < indices.size(); ++i) {
		shuffledImages[i] = images[indices[i]];
		shuffledLabels[i] = labels[indices[i]];
	}

	// Swap the shuffled data with the original data
	images.swap(shuffledImages);
	labels.swap(shuffledLabels);
}

vector<vector<vector<double>>> batchify(vector<vector<double>>* data, int batchSize) {
	vector<vector<vector<double>>> batches;
	int examples = data->size();
	if (examples % batchSize != 0) {
		printf("BATCHES NOT EVEN");
		// Throw an exception or return an empty vector
		throw std::invalid_argument("The number of examples is not evenly divisible by the batch size.");
	}
	int num_batches = examples / batchSize;
	int k = 0;

	for (int i = 0;i < num_batches;i++) {
		batches.push_back(vector<vector<double>>());
		for (int j = 0;j < batchSize;j++) {
			batches[i].push_back((*data)[k]);
			k++;
		}
	}
	return batches;
}