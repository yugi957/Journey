#include "../Headers/ImageProcessing.h"
#include "../../general.h"

vec3 getDims(vec3 a_dim, size_t k_size, size_t padding, size_t stride, size_t out_depth) {
	int height = floor(((a_dim.x + 2 * padding - k_size) / stride)) + 1;
	int width = floor(((a_dim.y + 2 * padding - k_size) / stride)) + 1;
	return vec3(height, width, out_depth);
}


float* convolve(vec3 a_dim, float* a, vec3 k_dim, int out_depth, float* kernel, int padding, int stride) {

	vec3 out_dim;
	int height = floor(((a_dim.x + 2 * padding - k_dim.x) / stride)) + 1;
	int width = floor(((a_dim.y + 2 * padding - k_dim.y) / stride)) + 1;

	out_dim = vec3(height, width, out_depth);

	float* b = new float[height * width * out_depth];
	int kernelSize = k_dim.x;
	int channels = a_dim.z;

	size_t a_size = a_dim.x * a_dim.y * a_dim.z;
	int end = (int)((k_dim.x / 2.0) + .5);
	int start = end - k_dim.x;
	int conv_row = 0 - start;
	float sum;

	for (int r = -padding, out_r = 0; out_r < height; r += stride, ++out_r) {
		for (int c = -padding, out_c = 0; out_c < width; c += stride, ++out_c) {
			for (int chan = 0;chan < channels;chan++) {
				sum = 0;
				for (int i = start;i < end;i++) {
					for (int j = start;j < end;j++) {
						//for (int chan = 0;chan < channels;chan++) {
						int a_id = ((r + i) * a_dim.y * a_dim.z) + ((c + j) * a_dim.z) + chan;
						//int k_id = ((i-start) * k_dim.y * k_dim.z) + ((j-start) * k_dim.z) + chan;
						int k_id = ((i - start) * k_dim.y * k_dim.z) + ((j - start) * k_dim.z);
						float term = 0;
						if (a_id >= 0 && a_id < a_size) {
							term = a[a_id] * kernel[k_id];
						}
						sum += term;
					}
				}
				b[out_r * width * channels + out_c * channels + chan] = sum;
			}
		}
	}
	return b;
}

float* convolve_volume(vec3 a_dim, float* a, vec3 k_dim, size_t out_depth, float* kernel, size_t padding, size_t stride) {

	vec3 out_dim;
	int height = floor(((a_dim.x + 2 * padding - k_dim.x) / stride)) + 1;
	int width = floor(((a_dim.y + 2 * padding - k_dim.y) / stride)) + 1;

	out_dim = vec3(height, width, out_depth);

	float* b = new float[height * width * out_depth];
	int kernelSize = k_dim.x;
	int channels = a_dim.z;

	size_t a_size = a_dim.x * a_dim.y * a_dim.z;
	int end = (int)((k_dim.x / 2.0) + .5);
	int start = end - k_dim.x;
	float sum;

	for (int out_chan = 0;out_chan < out_depth;out_chan++) {
		for (int r = -padding, out_r = 0; out_r < height; r += stride, ++out_r) {
			for (int c = -padding, out_c = 0; out_c < width; c += stride, ++out_c) {
				sum = 0;
				for (int i = start;i < end;i++) {
					for (int j = start;j < end;j++) {
						for (int chan = 0;chan < channels;chan++) {
							int a_id = ((r + i) * a_dim.y * a_dim.z) + ((c + j) * a_dim.z) + chan;
							int k_id = ((i - start) * k_dim.y * k_dim.z) + ((j - start) * k_dim.z) + chan;
							//int k_id = ((i - start) * k_dim.y * k_dim.z) + ((j - start) * k_dim.z);
							float term = 0;
							if (a_id >= 0 && a_id < a_size) {
								term = a[a_id] * kernel[k_id];
							}
							sum += term;
						}
					}
				}
				b[out_r * width * out_depth + out_c * out_depth + out_chan] = sum;
			}
		}
	}
	return b;
}

void convolve_volume(vec3 a_dim, vector<float>& a, vec3 k_dim, vector<vector<vector<vector<float>>>>& kernel, vec3 out_dim, vector<float>& out, size_t padding, size_t stride) {
	int height = out_dim.x;
	int width = out_dim.y;

	int kernelSize = k_dim.x;
	int channels = a_dim.z;

	size_t a_size = a_dim.x * a_dim.y * a_dim.z;
	int end = (int)((k_dim.x / 2.0) + .5);
	int start = end - k_dim.x;
	float sum;

	for (int out_chan = 0;out_chan < out_dim.z;out_chan++) {
		for (int r = -padding, out_r = 0; out_r < height; r += stride, ++out_r) {
			for (int c = -padding, out_c = 0; out_c < width; c += stride, ++out_c) {
				sum = 0;
				for (int i = 0;i < kernelSize;i++) {
					for (int j = 0;j < kernelSize;j++) {
						for (int chan = 0;chan < channels;chan++) {
							int a_id = ((r + i) * a_dim.y * a_dim.z) + ((c + j) * a_dim.z) + chan;
							//int k_id = ((i - start) * k_dim.y * k_dim.z) + ((j - start) * k_dim.z);
							float term = 0;
							if (a_id >= 0 && a_id < a_size) {
								term = a[a_id] * kernel[out_chan][i][j][chan];
							}
							sum += term;
						}
					}
				}
				out[out_r * width * out_dim.z + out_c * out_dim.z + out_chan] = sum;
			}
		}
	}
}

void backVolve(vec3 a_dim, vector<float>& terms, vec3 k_dim, vector<vector<vector<vector<float>>>>& kernel, vec3 out_dim, vector<float>& for_terms, size_t padding, size_t stride) {
	int height = out_dim.x;
	int width = out_dim.y;

	int kernelSize = k_dim.x;
	int channels = a_dim.z;

	size_t a_size = a_dim.x * a_dim.y * a_dim.z;
	int end = (int)((k_dim.x / 2.0) + .5);
	int start = end - k_dim.x;
	float sum;
	fill(terms.begin(), terms.end(), 0);

	for (int r = -padding, out_r = 0; out_r < height; r += stride, ++out_r) {
		for (int c = -padding, out_c = 0; out_c < width; c += stride, ++out_c) {
			sum = 0;
			for (int i = 0;i < kernelSize;i++) {
				for (int j = 0;j < kernelSize;j++) {
					for (int chan = 0;chan < channels;chan++) {
						int a_id = ((r + i) * a_dim.y * a_dim.z) + ((c + j) * a_dim.z) + chan;
						float term = 0;
						if (a_id >= 0 && a_id < a_size) {
							for (int out_chan = 0;out_chan < out_dim.z;out_chan++){
								terms[a_id] += kernel[out_chan][i][j][chan] * for_terms[out_r * width * out_dim.z + out_c * out_dim.z + out_chan];
							}
						}
					}
				}
			}
		}
	}
}

void backPool(vec3 a_dim, vector<float>& a, vector<float>& terms, vec3 out_dim, vector<float>& for_terms, size_t pool_size, size_t padding, size_t stride) {
	int height = out_dim.x;
	int width = out_dim.y;

	int kernelSize = pool_size;
	int channels = a_dim.z;

	size_t a_size = a_dim.x * a_dim.y * a_dim.z;
	int end = (int)((pool_size / 2.0) + .5);
	int start = end - pool_size;
	float max;
	fill(terms.begin(), terms.end(), 0);
	for (int r = -padding, out_r = 0; out_r < height; r += stride, ++out_r) {
		for (int c = -padding, out_c = 0; out_c < width; c += stride, ++out_c) {
			for (int chan = 0;chan < channels;chan++) {
				max = -INFINITY;
				int a_id;
				int id;
				for (int i = 0;i < kernelSize;i++) {
					for (int j = 0;j < kernelSize;j++) {
						a_id = ((r + i) * a_dim.y * a_dim.z) + ((c + j) * a_dim.z) + chan;
						float term = 0;
						if (a_id >= 0 && a_id < a_size) {
							term = a[a_id];
							if (term > max) {
								max = term;
								id = a_id;
							}
						}
					}
				}
				terms[id] = for_terms[out_r * width * a_dim.z + out_c * a_dim.z + chan];
			}
		}
	}
}

void gradVolve(vec3 a_dim, vector<float>& in_layer, vec3 k_dim, vector<vector<vector<vector<float>>>>& gradient, vec3 out_dim, vector<float>& out_terms, size_t padding, size_t stride) {
	int height = out_dim.x;
	int width = out_dim.y;

	int kernelSize = k_dim.x;
	int channels = a_dim.z;

	size_t a_size = a_dim.x * a_dim.y * a_dim.z;
	int end = (int)((k_dim.x / 2.0) + .5);
	int start = end - k_dim.x;
	float sum;

	for (int out_chan = 0;out_chan < out_dim.z;out_chan++) {
		for (int k = 0;k < k_dim.x;k++) {
			for (int j = 0;j < k_dim.y;j++) {
				for (int chan = 0;chan < k_dim.z;chan++) {
					gradient[out_chan][k][j][chan] = 0;
				}
			}
		}
		for (int r = -padding, out_r = 0; out_r < height; r += stride, ++out_r) {
			for (int c = -padding, out_c = 0; out_c < width; c += stride, ++out_c) {
				for (int i = 0;i < kernelSize;i++) {
					for (int j = 0;j < kernelSize;j++) {
						for (int chan = 0;chan < channels;chan++) {
							int a_id = ((r + i) * a_dim.y * a_dim.z) + ((c + j) * a_dim.z) + chan;
							if (a_id >= 0 && a_id < a_size) {
								 gradient[out_chan][i][j][chan] += in_layer[a_id] * out_terms[out_r * width * out_dim.z + out_c * out_dim.z + out_chan];
							}
						}
					}
				}
			}
		}
	}
}


float* max_pool(float* a, vec3 a_dim, size_t pool_size, size_t padding, size_t stride) {
	vec3 out_dim;
	int height = floor(((a_dim.x + 2 * padding - pool_size) / stride)) + 1;
	int width = floor(((a_dim.y + 2 * padding - pool_size) / stride)) + 1;

	out_dim = vec3(height, width, a_dim.z);

	float* b = new float[height * width * a_dim.z];
	int kernelSize = pool_size;
	int channels = a_dim.z;

	size_t a_size = a_dim.x * a_dim.y * a_dim.z;
	int end = (int)((pool_size / 2.0) + .5);
	int start = end - pool_size;
	float max;
	for (int r = -padding, out_r = 0; out_r < height; r += stride, ++out_r) {
		for (int c = -padding, out_c = 0; out_c < width; c += stride, ++out_c) {
			for (int chan = 0;chan < channels;chan++) {
				max = 0;
				for (int i = start;i < end;i++) {
					for (int j = start;j < end;j++) {
						int a_id = ((r + i) * a_dim.y * a_dim.z) + ((c + j) * a_dim.z) + chan;
						float term = 0;
						if (a_id >= 0 && a_id < a_size) {
							term = a[a_id];
						}
						if (term > max) max = term;
					}
				}
				b[out_r * width * a_dim.z + out_c * a_dim.z + chan] = max;
			}
		}
	}
	return b;
}

void max_pool(vec3 a_dim, vector<float>& a, vec3 out_dim, vector<float>& out, size_t pool_size, size_t padding, size_t stride) {
	int height = out_dim.x;
	int width = out_dim.y;

	int kernelSize = pool_size;
	int channels = a_dim.z;

	size_t a_size = a_dim.x * a_dim.y * a_dim.z;
	int end = (int)((pool_size / 2.0) + .5);
	int start = end - pool_size;
	float max;
	for (int r = -padding, out_r = 0; out_r < height; r += stride, ++out_r) {
		for (int c = -padding, out_c = 0; out_c < width; c += stride, ++out_c) {
			for (int chan = 0;chan < channels;chan++) {
				max = 0;
				for (int i = 0;i < pool_size;i++) {
					for (int j = 0;j < pool_size;j++) {
						int a_id = ((r + i) * a_dim.y * a_dim.z) + ((c + j) * a_dim.z) + chan;
						float term = 0;
						if (a_id >= 0 && a_id < a_size) {
							term = a[a_id];
						}
						if (term > max) max = term;
					}
				}
				out[out_r * width * a_dim.z + out_c * a_dim.z + chan] = max;
			}
		}
	}
}