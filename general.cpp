#include "general.h"

void xavier_init(vector<vector<float>>& weights, int input_size, int output_size) {
	// Define a random number generator
	default_random_engine generator;
	normal_distribution<float> distribution(0.0, 1.0);

	// Resize the weights matrix to the appropriate size
	weights.resize(output_size, vector<float>(input_size));

	// Compute the scaling factor
	float scaling_factor = sqrt(6.0 / (input_size + output_size));

	// Fill the weights matrix with random values
	for (int i = 0; i < output_size; i++) {
		for (int j = 0; j < input_size; j++) {
			weights[i][j] = distribution(generator) * scaling_factor;
		}
	}
}


void he_init(vector<vector<float>>& weights, int input_size, int output_size) {
	// Define a random number generator
	default_random_engine generator;
	normal_distribution<float> distribution(0.0, 1.0);

	// Resize the weights matrix to the appropriate size
	weights.resize(output_size, vector<float>(input_size));

	// Compute the scaling factor
	float scaling_factor = sqrt(2.0 / input_size);

	// Fill the weights matrix with random values
	for (int i = 0; i < output_size; i++) {
		for (int j = 0; j < input_size; j++) {
			weights[i][j] = distribution(generator) * scaling_factor;
		}
	}
}

void xavier_init_conv(vector<vector<vector<vector<float>>>>& conv_weights, int input_channels, int output_channels, int kernel_size) {
	// Define a random number generator
	default_random_engine generator;
	normal_distribution<float> distribution(0.0, 1.0);

	// Compute the scaling factor
	float scaling_factor = sqrt(6.0 / (input_channels * kernel_size * kernel_size + output_channels));

	// Fill the weights tensor with random values
	for (int oc = 0; oc < output_channels; oc++) {
		for (int ks1 = 0; ks1 < kernel_size; ks1++) {
			for (int ks2 = 0; ks2 < kernel_size; ks2++) {
				for (int ic = 0; ic < input_channels; ic++) {
					conv_weights[oc][ks1][ks2][ic] = distribution(generator) * scaling_factor;
				}
			}
		}
	}
}

void he_init_conv(vector<vector<vector<vector<float>>>>& conv_weights, int input_channels, int output_channels, int kernel_size) {
	// Define a random number generator
	random_device rd;
	mt19937 generator(rd());
	normal_distribution<float> distribution(0.0, 1.0);

	// Compute the scaling factor
	float scaling_factor = sqrt(2.0 / (input_channels * kernel_size * kernel_size));

	// Fill the weights tensor with random values
	for (int oc = 0; oc < output_channels; oc++) {
		for (int ks1 = 0; ks1 < kernel_size; ks1++) {
			for (int ks2 = 0; ks2 < kernel_size; ks2++) {
				for (int ic = 0; ic < input_channels; ic++) {
					conv_weights[oc][ks1][ks2][ic] = distribution(generator) * scaling_factor;
				}
			}
		}
	}
}

float frand() {
	return (2.0 * (float)rand() / RAND_MAX) - 1.0;
}

float getSum(vector<float> x) {
	float sum = 0;
	for (int i = 0;i < x.size();i++) {
		sum += x[i];
	}
	return sum;
}

float max(vector<float> x) {
	float max = 0;
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
void printArray(float* arr, int size) {
	printf("[");
	for (int i = 0;i < size;i++) {
		printf("% f, ", arr[i]);
	}
	printf("]\n");
}

void printExecution(char* s, clock_t start, clock_t end) {
	printf("%s: %4.7f\n", s, (float)((float)(end - start) / CLOCKS_PER_SEC));
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

void compare_arrays(float* a, float* b, int size) {
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

int getSum(float* a, int size) {
	float sum = 0;
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

void average3D(vector<vector<vector<float>>>* a, vector<vector<float>>* b) {
	float sum;
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

void compare3D(vector<vector<vector<float>>> a, vector<vector<vector<float>>> b) {
	for (int i = 0;i < a.size();i++) {
		for (int j = 0;j < a[i].size();j++) {
			for (int k = 0;k < a[i][j].size();k++) {
				if (a[i][j][k] - b[i][j][k] < -1 * .000001 || a[i][j][k] - b[i][j][k] > .000001) {
					printf("%f :: %f\n", a[i][j][k], b[i][j][k]);
					printf("\nARRAYS ARE DIFFERENT\n\n");
					return;
				}
			}
		}
	}
	printf("SUCCESS Arrays are the same\n");
}

void compareHtoConvWeight(vector<vector<vector<float>>> a, vector<vector<vector<vector<vector<float>>>>> b) {
	for (int i = 0;i < a.size();i++) {
		for (int j = 0;j < a[i].size();j++) {
			for (int k = 0;k < a[i][j].size();k++) {
				if (a[i][j][k] - b[i][0][0][j][k] < -1 * .000001 || a[i][j][k] - b[i][0][0][j][k] > .000001) {
					printf("%f :: %f\n", a[i][j][k], b[i][0][0][j][k]);
					printf("\nARRAYS ARE DIFFERENT\n\n");
					return;
				}
			}
		}
	}
	printf("SUCCESS Arrays are the same\n");
}

bool compare2D(vector<vector<float>> a, vector<vector<float>> b) {
	for (int i = 0;i < a.size();i++) {
		for (int j = 0;j < a[i].size();j++) {
			if (a[i][j] - b[i][j] < -1 * .000001 || a[i][j] - b[i][j] > .000001) {
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

void compare5D(vector<vector<vector<vector<vector<float>>>>> a, vector<vector<vector<vector<vector<float>>>>> b) {
	for (int i = 0;i < a.size();i++) {
		for (int j = 0;j < a[i].size();j++) {
			for (int k = 0;k < a[i][j].size();k++) {
				for (int l = 0;l < a[i][j][k].size();l++) {
					for (int m = 0;m < a[i][j][k][l].size();m++) {
						if (a[i][j][k][l][m] - b[i][j][k][l][m] < -1 * .000001 || a[i][j][k][l][m] - b[i][j][k][l][m] > .000001) {
							printf("%f :: %f\n", a[i][j][k][l][m], b[i][j][k][l][m]);
							printf("\nARRAYS ARE DIFFERENT\n\n");
							return;
						}
					}
				}
			}
		}
	}
	printf("SUCCESS Arrays are the same\n");
}

void shuffleData(vector<vector<float>>& images, vector<vector<float>>& labels) {
	// Seed with a real random value, if available
	random_device rd;

	// Create a random number generator
	mt19937 g(rd());

	// Create a vector of indices
	vector<size_t> indices(images.size());
	iota(indices.begin(), indices.end(), 0);

	// Shuffle the indices
	shuffle(indices.begin(), indices.end(), g);

	// Create temporary vectors to hold the shuffled data
	vector<vector<float>> shuffledImages(images.size());
	vector<vector<float>> shuffledLabels(labels.size());

	// Rearrange the data according to the shuffled indices
	for (size_t i = 0; i < indices.size(); ++i) {
		shuffledImages[i] = images[indices[i]];
		shuffledLabels[i] = labels[indices[i]];
	}

	// Swap the shuffled data with the original data
	images.swap(shuffledImages);
	labels.swap(shuffledLabels);
}

//vector<vector<vector<float>>> batchify(vector<vector<float>>* data, int batchSize) {
//	vector<vector<vector<float>>> batches;
//	int examples = data->size();
//	if (examples % batchSize != 0) {
//		printf("BATCHES NOT EVEN");
//		// Throw an exception or return an empty vector
//		throw invalid_argument("The number of examples is not evenly divisible by the batch size.");
//	}
//	int num_batches = examples / batchSize;
//	int k = 0;
//
//	for (int i = 0;i < num_batches;i++) {
//		batches.push_back(vector<vector<float>>());
//		for (int j = 0;j < batchSize;j++) {
//			batches[i].push_back((*data)[k]);
//			k++;
//		}
//	}
//	return batches;
//}

//template <typename T>
//vector<T> flatten4D(vector<vector<vector<vector<T>>>>& to_flatten) {
//	vector<T> flattened = {};
//	for (auto a : to_flatten)
//		for (auto b : a)
//			for (auto c : b)
//				for (auto val : c)
//					flattened.push_back(val);
//	return flattened;
//}

vector<vector<float>> autoencode(vector<vector<float>> set, int size) {
	vector<float> temp(size, 0);
	vector<vector<float>> res(set.size(), vector<float>(size));
	for (int i = 0;i < set.size();i++) {
		temp[set[i][0]] = 1;
		res[i] = temp;
		temp[set[i][0]] = 0;
	}
	return res;
}


void write4D(vector<vector<vector<vector<float>>>> vec4d) {
	ofstream file("weights.txt");

	for (const auto& dim1 : vec4d) {
		for (const auto& dim2 : dim1) {
			bool first3 = true;
			for (const auto& dim3 : dim2) {
				if (!first3) file << ";";
				first3 = false;
				bool first4 = true;
				for (const auto& val : dim3) {
					if (!first4) file << ", ";
					first4 = false;
					file << val;
				}
			}
			file << "\n"; // Separate the second dimension
		}
		file << "\n"; // Separate the top level vectors
	}
	file.close();
}

void write2D(vector<vector<float>> vec2d) {
	ofstream file("outputs.txt");

	for (const auto& row : vec2d) {
		bool first = true;
		for (const auto& val : row) {
			if (!first) file << ", ";
			file << val;
			first = false;
		}
		file << "\n";
	}

	file.close();
}