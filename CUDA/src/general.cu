#include "../Headers/general.cuh"

inline void gpuAssert(cudaError code, const char* file, int line) {
	bool abort = true;
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

void cudaAllocate2dOffVectorHostRef(float*** d_a, vector<vector<float>> h_a) {
	bool vocal = false;
	*d_a = new float* [h_a.size()];
	for (int i = 0;i < h_a.size();i++) {
		if (vocal) printf("size %d: %d\n", i, h_a[i].size());
		cudaMalloc((void**)&(*d_a)[i], sizeof(float) * h_a[i].size());
		cudaMemcpy((*d_a)[i], &(h_a)[i][0], sizeof(float) * h_a[i].size(), cudaMemcpyHostToDevice);
	}
	if (vocal) printf("\n");
}

void cudaFree2dHostRef(float*** d_a, int size) {
	for (int i = 0;i < size;i++) {
		cudaFree((*d_a)[i]);
	}
}

void cudaMemCopy2dOffVectorHostRef(float*** d_a, vector<vector<float>> h_a) {
	bool vocal = false;
	for (int i = 0;i < h_a.size();i++) {
		if (vocal) printf("size %d: %d\n", i, h_a[i].size());
		//cudaMalloc((void**)&(*d_a)[i], sizeof(float) * h_a[i].size());
		cudaMemcpy((*d_a)[i], &(h_a)[i][0], sizeof(float) * h_a[i].size(), cudaMemcpyHostToDevice);
	}
	if (vocal) printf("\n");
}

void cudaAllocateFull2dOffVectorHostRef(float*** d_a, vector<vector<float>> h_a, int batchSize) {
	*d_a = new float* [batchSize];
	int size = 0;
	for (int i = 0;i < h_a.size();i++) {
		size += h_a[i].size();
	}
	float* a = new float[size];
	int c = 0;
	for (int i = 0;i < h_a.size();i++) {
		for (int j = 0;j < h_a[i].size();j++) {
			a[c] = h_a[i][j];
			c++;
		}
	}
	if (c != size) printf("WHA HAPPEN\n");
	for (int i = 0;i < batchSize;i++) {
		cudaMalloc((void**)&(*d_a)[i], sizeof(float) * size);
		cudaMemcpy((*d_a)[i], a, sizeof(float) * size, cudaMemcpyHostToDevice);
	}
}

void cudaAllocate3dOffVectorHostRef(float*** d_a, vector<vector<vector<float>>> h_a) {
	*d_a = new float* [h_a.size()];
	for (int i = 0;i < h_a.size();i++) {
		cudaAllocate2dOffVector(&(*d_a)[i], h_a[i]);
	}
}

void cudaAllocate2dOffVector(float** d_a, vector<vector<float>> h_inputs, int** lengths) {
	int size = 0;
	int* lens = new int[h_inputs.size()];
	for (int i = 0;i < h_inputs.size();i++) {
		size += h_inputs[0].size();
		lens[i] = h_inputs[0].size();
	}
	*lengths = lens;
	cudaMalloc((void**)&(*d_a), size * sizeof(float));
	float* h_a = new float[size];
	int c = 0;
	for (int i = 0;i < h_inputs.size();i++) {
		for (int j = 0;j < h_inputs[0].size();j++) {
			h_a[c] = h_inputs[i][j];
			c++;
		}
	}
	cudaMemcpy(*d_a, h_a, size * sizeof(float), cudaMemcpyHostToDevice);
	free(lens);
	free(h_a);
}

void cudaAllocate2dOffVector(float** d_a, vector<vector<float>> h_inputs) {
	int size = 0;
	for (int i = 0;i < h_inputs.size();i++) {
		size += h_inputs[i].size();
	}
	cudaMalloc((void**)&(*d_a), size * sizeof(float));
	float* h_a = new float[size];
	int c = 0;
	for (int i = 0;i < h_inputs.size();i++) {
		for (int j = 0;j < h_inputs[i].size();j++) {
			h_a[c] = h_inputs[i][j];
			c++;
		}
	}
	cudaMemcpy(*d_a, h_a, size * sizeof(float), cudaMemcpyHostToDevice);
	free(h_a);
}


void cudaAllocate3dOffVector(float** d_a, vector<vector<vector<float>>> h_inputs) {
	int size = 0;
	for (int i = 0;i < h_inputs.size();i++) {
		for (int j = 0;j < h_inputs[i].size();j++) {
			size += h_inputs[i][j].size();
		}
	}
	cudaMalloc((void**)&(*d_a), size * sizeof(float));
	float* h_a = new float[size];
	int c = 0;
	for (int i = 0;i < h_inputs.size();i++) {
		for (int j = 0;j < h_inputs[i].size();j++) {
			for (int k = 0;k < h_inputs[i][j].size();k++) {
				h_a[c] = h_inputs[i][j][k];
				c++;
			}
		}
	}
	cudaMemcpy(*d_a, h_a, size * sizeof(float), cudaMemcpyHostToDevice);
	free(h_a);
}

void cudaMemcpy2dOffVector(float** d_a, vector<vector<float>> h_inputs) {
	int size = 0;
	for (int i = 0;i < h_inputs.size();i++) {
		size += h_inputs[i].size();
	}
	float* h_a = new float[size];
	int c = 0;
	for (int i = 0;i < h_inputs.size();i++) {
		for (int j = 0;j < h_inputs[i].size();j++) {
			h_a[c] = h_inputs[i][j];
			c++;
		}
	}
	cudaMemcpy(*d_a, h_a, size * sizeof(float), cudaMemcpyHostToDevice);
	free(h_a);
}

void cudaMemcpy3dOffVector(float** d_a, vector<vector<vector<float>>> h_inputs) {
	int size = 0;
	for (int i = 0;i < h_inputs.size();i++) {
		for (int j = 0;j < h_inputs[i].size();j++) {
			size += h_inputs[i][j].size();
		}
	}
	float* h_a = new float[size];
	int c = 0;
	for (int i = 0;i < h_inputs.size();i++) {
		for (int j = 0;j < h_inputs[i].size();j++) {
			for (int k = 0;k < h_inputs[i][j].size();k++) {
				h_a[c] = h_inputs[i][j][k];
				c++;
			}
		}
	}
	cudaMemcpy(*d_a, h_a, size * sizeof(float), cudaMemcpyHostToDevice);
	free(h_a);
}

void cudaMemcpy3dOffVectorHostRef(float*** d_a, vector<vector<vector<float>>> h_a) {
	for (int i = 0;i < h_a.size();i++) {
		cudaMemcpy2dOffVector((*d_a) + i, h_a[i]);
	}
}

vector<vector<float>> cudaCopy2dBackToVector(float** d_a, vector<int> lengths) {
	vector<int> offsets;
	offsets.push_back(0);
	for (int i = 1;i < lengths.size();i++) {
		offsets.push_back(offsets[i - 1] + lengths[i - 1]);
	}
	vector<vector<float>> a;
	int size = 0;
	for (int i = 0;i < lengths.size();i++) {
		size += lengths[i];
	}
	float* h_a = new float[size];
	cudaMemcpy(h_a, *d_a, size * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 0;i < lengths.size();i++) {
		a.push_back(vector<float>());
		for (int j = 0;j < lengths[i];j++) {
			a[i].push_back(h_a[offsets[i] + j]);
		}
	}

	free(h_a);
	return a;
}

vector<vector<vector<float>>> cudaCopy3dBackToVector(float** d_a, vector<vector<int>> lengths) {
	vector<int> offsets;
	offsets.push_back(0);
	for (int i = 1;i < lengths.size();i++) {
		int sum = 0;
		for (int j = 0;j < lengths[i].size();j++) {
			sum += lengths[i - 1][j];
		}
		offsets.push_back(offsets[i - 1] + sum);
	}
	vector<vector<vector<float>>> a;
	int size = 0;
	for (int i = 0;i < lengths.size();i++) {
		float sum = 0;
		for (int j = 0;j < lengths[i].size();j++) {
			sum += lengths[i][j];
		}
		size += sum;
	}
	float* h_a = new float[size];
	cudaMemcpy(h_a, *d_a, size * sizeof(float), cudaMemcpyDeviceToHost);
	int c = 0;
	for (int i = 0;i < lengths.size();i++) {
		a.push_back(vector<vector<float>>());
		for (int j = 0;j < lengths[i].size();j++) {
			a[i].push_back(vector<float>());
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

vector<float> cudaCopy2dBackTo1dVector(float** d_a, vector<int> lengths) {
	vector<int> offsets;
	offsets.push_back(0);
	for (int i = 1;i < lengths.size() - 1;i++) {
		offsets.push_back(offsets[i - 1] + lengths[i - 1]);
	}
	vector<vector<float>> a;
	int size = 0;
	for (int i = 0;i < lengths.size();i++) {
		size += lengths[i];
	}
	vector<float> out(size, 0.0);
	cudaMemcpy(&out[0], *d_a, size * sizeof(float), cudaMemcpyDeviceToHost);
	return out;
}

vector<vector<float>> cudaCopy2dBackToVectorHref(float** d_a, vector<int> lengths) {
	vector<vector<float>> a = vector<vector<float>>(lengths.size());
	for (int i = 0;i < lengths.size();i++) {
		a[i] = vector<float>(lengths[i]);
		gpuErrorchk(cudaMemcpy(&a[i][0], &(*d_a)[i], sizeof(float) * lengths[i], cudaMemcpyDeviceToHost));
	}
	return a;
}

vector<vector<vector<float>>> cudaCopy3dBackToVectorHref(float*** d_a, vector<vector<int>> lengths) {
	vector<vector<vector<float>>> a;
	float** h_a_href = new float* [lengths.size()];
	int size = 0;
	for (int i = 0;i < lengths.size();i++) {
		float sum = 0;
		for (int j = 0;j < lengths[i].size();j++) {
			sum += lengths[i][j];
		}
		h_a_href[i] = new float[sum];
		cudaMemcpy(h_a_href[i], (*d_a)[i], sizeof(float) * sum, cudaMemcpyDeviceToHost);
	}

	for (int i = 0;i < lengths.size();i++) {
		a.push_back(vector<vector<float>>());
		int c = 0;
		for (int j = 0;j < lengths[i].size();j++) {
			a[i].push_back(vector<float>());
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

vector<vector<float>> cudaCopyBatchBackToVectorHref(float** d_a, int size, int batchSize) {
	vector<vector<float>> out(batchSize, vector<float>(size));
	for (int i = 0;i < batchSize;i++) {
		cudaMemcpy(&out[i][0], &(*d_a)[i * size], sizeof(float) * size, cudaMemcpyDeviceToHost);
		//printf("COPYING::: %f\n", out[i][0]);
	}
	return out;
}

float*** createBatches(float** hr_a, int batchSize, int examples, int size) {
	int batchNum = examples / batchSize;
	float*** batches = new float** [batchNum];

	int c = 0;
	for (int i = 0;i < batchNum;i++) {
		batches[i] = new float* [batchSize];
		for (int j = 0;j < batchSize;j++) {
			batches[i][j] = hr_a[c];
			c++;
		}
	}
	return batches;
}

vector<vector<float>> batchify(vector<vector<float>>* data, int batchSize) {
	int examples = data->size();
	if (examples % batchSize != 0) {
		printf("BATCHES NOT EVEN");
		// Throw an exception or return an empty vector
		throw std::invalid_argument("The number of examples is not evenly divisible by the batch size.");
	}

	int num_batches = examples / batchSize;
	int k = 0;
	vector<vector<float>> batches;

	for (int i = 0;i < num_batches;i++) {
		batches.push_back(vector<float>());
		for (int j = i * batchSize;j < (i + 1) * batchSize;j++) {
			for (int k = 0;k < (*data)[j].size();k++) {
				batches[i].push_back((*data)[j][k]);
			}
		}
	}
	return batches;
}