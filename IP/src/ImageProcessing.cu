#include "../Headers/ImageProcessing.cuh"


dim3 getDims(dim3 a_dim, size_t k_size, size_t padding, size_t stride, size_t out_depth) {
	int height = floor(((a_dim.x + 2 * padding - k_size) / stride)) + 1;
	int width = floor(((a_dim.y + 2 * padding - k_size) / stride)) + 1;
	return dim3(height, width, out_depth);
}


float* convolve(dim3 a_dim, float* a, dim3 k_dim, int out_depth, float* kernel, int padding, int stride) {

	dim3 out_dim;
	int height = floor(((a_dim.x + 2 * padding - k_dim.x) / stride)) + 1;
	int width = floor(((a_dim.y + 2 * padding - k_dim.y) / stride)) + 1;

	out_dim = dim3(height, width, out_depth);

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

float* convolve_volume(dim3 a_dim, float* a, dim3 k_dim, size_t out_depth, float* kernel, size_t padding, size_t stride) {

	dim3 out_dim;
	int height = floor(((a_dim.x + 2 * padding - k_dim.x) / stride)) + 1;
	int width = floor(((a_dim.y + 2 * padding - k_dim.y) / stride)) + 1;

	out_dim = dim3(height, width, out_depth);

	float* b = new float[height * width * out_depth];
	int kernelSize = k_dim.x;
	int channels = a_dim.z;

	size_t a_size = a_dim.x * a_dim.y * a_dim.z;
	int end = (int)((k_dim.x / 2.0) + .5);
	int start = end - k_dim.x;
	int conv_row = 0 - start;
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
				b[out_r * width * out_depth + out_c * out_depth + out_depth] = sum;
			}
		}
	}
	return b;
}

float* max_pool(float* a, dim3 a_dim, size_t pool_size, size_t padding, size_t stride) {
	dim3 out_dim;
	int height = floor(((a_dim.x + 2 * padding - pool_size) / stride)) + 1;
	int width = floor(((a_dim.y + 2 * padding - pool_size) / stride)) + 1;

	out_dim = dim3(height, width, a_dim.z);

	float* b = new float[height * width * a_dim.z];
	int kernelSize = pool_size;
	int channels = a_dim.z;

	size_t a_size = a_dim.x * a_dim.y * a_dim.z;
	int end = (int)((pool_size / 2.0) + .5);
	int start = end - pool_size;
	int conv_row = 0 - start;
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