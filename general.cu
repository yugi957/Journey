#include "general.cuh"


void cudaAllocate2dOffVectorHostRef(double*** d_a, vector<vector<double>> h_a, bool vocal) {
	*d_a = new double* [h_a.size()];
	for (int i = 0;i < h_a.size();i++) {
		if (vocal) printf("size %d: %d\n", i, h_a[i].size());
		cudaMalloc((void**)&(*d_a)[i], sizeof(double) * h_a[i].size());
		cudaMemcpy((*d_a)[i], &(h_a)[i][0], sizeof(double) * h_a[i].size(), cudaMemcpyHostToDevice);
	}
	if (vocal) printf("\n");
}

void cudaAllocateFull2dOffVectorHostRef(double*** d_a, vector<vector<double>> h_a, int batchSize) {
	*d_a = new double* [batchSize];
	int size = 0;
	for (int i = 0;i < h_a.size();i++) {
		size += h_a[i].size();
	}
	double* a = new double[size];
	int c = 0;
	for (int i = 0;i < h_a.size();i++) {
		for (int j = 0;j < h_a[i].size();j++) {
			a[c] = h_a[i][j];
			c++;
		}
	}
	if (c != size) printf("WHA HAPPEN\n");
	for (int i = 0;i < batchSize;i++) {
		cudaMalloc((void**)&(*d_a)[i], sizeof(double) * size);
		cudaMemcpy((*d_a)[i], a, sizeof(double) * size, cudaMemcpyHostToDevice);
	}
}

void cudaAllocate3dOffVectorHostRef(double*** d_a, vector<vector<vector<double>>> h_a) {
	*d_a = new double* [h_a.size()];
	for (int i = 0;i < h_a.size();i++) {
		cudaAllocate2dOffVector(&(*d_a)[i], h_a[i]);
	}
}

void cudaAllocate2dOffVector(double** d_a, vector<vector<double>> h_inputs, int** lengths) {
	int size = 0;
	int* lens = new int[h_inputs.size()];
	for (int i = 0;i < h_inputs.size();i++) {
		size += h_inputs[0].size();
		lens[i] = h_inputs[0].size();
	}
	*lengths = lens;
	cudaMalloc((void**)&(*d_a), size * sizeof(double));
	double* h_a = new double[size];
	int c = 0;
	for (int i = 0;i < h_inputs.size();i++) {
		for (int j = 0;j < h_inputs[0].size();j++) {
			h_a[c] = h_inputs[i][j];
			c++;
		}
	}
	cudaMemcpy(*d_a, h_a, size * sizeof(double), cudaMemcpyHostToDevice);
	free(lens);
	free(h_a);
}

void cudaAllocate2dOffVector(double** d_a, vector<vector<double>> h_inputs) {
	int size = 0;
	for (int i = 0;i < h_inputs.size();i++) {
		size += h_inputs[i].size();
	}
	cudaMalloc((void**)&(*d_a), size * sizeof(double));
	double* h_a = new double[size];
	int c = 0;
	for (int i = 0;i < h_inputs.size();i++) {
		for (int j = 0;j < h_inputs[i].size();j++) {
			h_a[c] = h_inputs[i][j];
			c++;
		}
	}
	cudaMemcpy(*d_a, h_a, size * sizeof(double), cudaMemcpyHostToDevice);
	free(h_a);
}


void cudaAllocate3dOffVector(double** d_a, vector<vector<vector<double>>> h_inputs) {
	int size = 0;
	for (int i = 0;i < h_inputs.size();i++) {
		for (int j = 0;j < h_inputs[i].size();j++) {
			size += h_inputs[i][j].size();
		}
	}
	cudaMalloc((void**)&(*d_a), size * sizeof(double));
	double* h_a = new double[size];
	int c = 0;
	for (int i = 0;i < h_inputs.size();i++) {
		for (int j = 0;j < h_inputs[i].size();j++) {
			for (int k = 0;k < h_inputs[i][j].size();k++) {
				h_a[c] = h_inputs[i][j][k];
				c++;
			}
		}
	}
	cudaMemcpy(*d_a, h_a, size * sizeof(double), cudaMemcpyHostToDevice);
	free(h_a);
}

void cudaMemcpy2dOffVector(double** d_a, vector<vector<double>> h_inputs) {
	int size = 0;
	for (int i = 0;i < h_inputs.size();i++) {
		size += h_inputs[i].size();
	}
	double* h_a = new double[size];
	int c = 0;
	for (int i = 0;i < h_inputs.size();i++) {
		for (int j = 0;j < h_inputs[i].size();j++) {
			h_a[c] = h_inputs[i][j];
			c++;
		}
	}
	cudaMemcpy(*d_a, h_a, size * sizeof(double), cudaMemcpyHostToDevice);
	free(h_a);
}

void cudaMemcpy3dOffVector(double** d_a, vector<vector<vector<double>>> h_inputs) {
	int size = 0;
	for (int i = 0;i < h_inputs.size();i++) {
		for (int j = 0;j < h_inputs[i].size();j++) {
			size += h_inputs[i][j].size();
		}
	}
	double* h_a = new double[size];
	int c = 0;
	for (int i = 0;i < h_inputs.size();i++) {
		for (int j = 0;j < h_inputs[i].size();j++) {
			for (int k = 0;k < h_inputs[i][j].size();k++) {
				h_a[c] = h_inputs[i][j][k];
				c++;
			}
		}
	}
	cudaMemcpy(*d_a, h_a, size * sizeof(double), cudaMemcpyHostToDevice);
	free(h_a);
}

void cudaMemcpy3dOffVectorHostRef(double*** d_a, vector<vector<vector<double>>> h_a) {
	for (int i = 0;i < h_a.size();i++) {
		cudaMemcpy2dOffVector((*d_a)+i, h_a[i]);
	}
}

vector<vector<double>> cudaCopy2dBackToVector(double** d_a, vector<int> lengths) {
	vector<int> offsets;
	offsets.push_back(0);
	for (int i = 1;i < lengths.size();i++) {
		offsets.push_back(offsets[i - 1] + lengths[i - 1]);
	}
	vector<vector<double>> a;
	int size = 0;
	for (int i = 0;i < lengths.size();i++) {
		size += lengths[i];
	}
	double* h_a = new double[size];
	cudaMemcpy(h_a, *d_a, size * sizeof(double), cudaMemcpyDeviceToHost);

	for (int i = 0;i < lengths.size();i++) {
		a.push_back(vector<double>());
		for (int j = 0;j < lengths[i];j++) {
			a[i].push_back(h_a[offsets[i] + j]);
		}
	}

	free(h_a);
	return a;
}

vector<vector<vector<double>>> cudaCopy3dBackToVector(double** d_a, vector<vector<int>> lengths) {
	vector<int> offsets;
	offsets.push_back(0);
	for (int i = 1;i < lengths.size();i++) {
		int sum = 0;
		for (int j = 0;j < lengths[i].size();j++) {
			sum += lengths[i-1][j];
		}
		offsets.push_back(offsets[i - 1] + sum);
	}
	vector<vector<vector<double>>> a;
	int size = 0;
	for (int i = 0;i < lengths.size();i++) {
		double sum = 0;
		for (int j = 0;j < lengths[i].size();j++) {
			sum += lengths[i][j];
		}
		size += sum;
	}
	double* h_a = new double[size];
	cudaMemcpy(h_a, *d_a, size * sizeof(double), cudaMemcpyDeviceToHost);
	int c = 0;
	for (int i = 0;i < lengths.size();i++) {
		a.push_back(vector<vector<double>>());
		for (int j = 0;j < lengths[i].size();j++) {
			a[i].push_back(vector<double>());
			for (int k = 0;k < lengths[i][j];k++) {
				//a[i][j].push_back(h_a[offsets[i] + j * lengths[i][j] + k]);
				a[i][j].push_back(h_a[c]);
				c++;
			}
		}
	}

	free(h_a);
	return a;
}

vector<double> cudaCopy2dBackTo1dVector(double** d_a, vector<int> lengths) {
	vector<int> offsets;
	offsets.push_back(0);
	for (int i = 1;i < lengths.size() - 1;i++) {
		offsets.push_back(offsets[i - 1] + lengths[i - 1]);
	}
	vector<vector<double>> a;
	int size = 0;
	for (int i = 0;i < lengths.size();i++) {
		size += lengths[i];
	}
	vector<double> out(size, 0.0);
	cudaMemcpy(&out[0], *d_a, size * sizeof(double), cudaMemcpyDeviceToHost);
	return out;
}

vector<vector<double>> cudaCopy2dBackToVectorHref(double** d_a, vector<int> lengths) {
	vector<vector<double>> a = vector<vector<double>>(lengths.size());
	for (int i = 0;i < lengths.size();i++) {
		a[i] = vector<double>(lengths[i]);
		cudaMemcpy(&a[i][0], &(*d_a)[i], sizeof(double) * lengths[i], cudaMemcpyDeviceToHost);
	}
	return a;
}

vector<vector<vector<double>>> cudaCopy3dBackToVectorHref(double*** d_a, vector<vector<int>> lengths) {
	vector<vector<vector<double>>> a;
	double** h_a_href = new double* [lengths.size()];
	int size = 0;
	for (int i = 0;i < lengths.size();i++) {
		double sum = 0;
		for (int j = 0;j < lengths[i].size();j++) {
			sum += lengths[i][j];
		}
		h_a_href[i] = new double[sum];
		cudaMemcpy(h_a_href[i], (*d_a)[i], sizeof(double) * sum, cudaMemcpyDeviceToHost);
	}

	for (int i = 0;i < lengths.size();i++) {
		a.push_back(vector<vector<double>>());
		int c = 0;
		for (int j = 0;j < lengths[i].size();j++) {
			a[i].push_back(vector<double>());
			for (int k = 0;k < lengths[i][j];k++) {
				a[i][j].push_back(h_a_href[i][c]);
				c++;
			}
		}
	}

	for (int i = 0;i < lengths.size();i++) free(h_a_href[i]);
	free(h_a_href);
	return a;

}

vector<vector<double>> cudaCopyBatchBackToVectorHref(double** d_a, int size, int batchSize) {
	vector<vector<double>> out(batchSize, vector<double>(size));
	for (int i = 0;i < batchSize;i++) {
		cudaMemcpy(&out[i][0], &(*d_a)[i * size], sizeof(double) * size, cudaMemcpyDeviceToHost);
		//printf("COPYING::: %f\n", out[i][0]);
	}
	return out;
}

double*** createBatches(double** hr_a, int batchSize, int examples, int size) {
	int batchNum = examples / batchSize;
	double*** batches = new double** [batchNum];

	int c = 0;
	for (int i = 0;i < batchNum;i++) {
		batches[i] = new double* [batchSize];
		for (int j = 0;j < batchSize;j++) {
			batches[i][j] = hr_a[c];
			c++;
		}
	}
	return batches;
}

vector<vector<double>> batchify(vector<vector<double>>* data, int batchSize) {
	int examples = data->size();
	if (examples % batchSize != 0) {
		printf("BATCHES NOT EVEN");
		// Throw an exception or return an empty vector
		throw std::invalid_argument("The number of examples is not evenly divisible by the batch size.");
	}

	int num_batches = examples / batchSize;
	int k = 0;
	vector<vector<double>> batches;

	for (int i = 0;i < num_batches;i++) {
		batches.push_back(vector<double>());
		for (int j = i * batchSize;j < (i+1)*batchSize;j++) {
			for (int k = 0;k < (*data)[j].size();k++) {
				batches[i].push_back((*data)[j][k]);
			}
		}
	}
	return batches;
}