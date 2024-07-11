#include "../general.h"
#include "../CUDA/Headers/general.cuh"
#include <opencv2/opencv.hpp>
#include "Headers/ImageProcessing.cuh"
#include "Headers/kernels.cuh"
#include <fstream>
using namespace cv;

int main() {

	string image_path = "C:/Atul/cuDNN/images/";
	string full_path = image_path + "mountain.jpg";

	ifstream file(full_path);
	if (!file) {
		cout << "File does not exist: " << full_path << endl;
		//return -1;
	}
	file.close();
	Mat img = imread(image_path + "view.jpg");
	int screen_width = 1920; // Example screen width
	int screen_height = 1080; // Example screen height

	// Resize image if it is larger than screen dimensions
	if (img.cols > screen_width || img.rows > screen_height) {
		double aspect_ratio = (double)img.cols / img.rows;
		if (img.cols > img.rows) {
			int new_height = (screen_width / aspect_ratio) * .9;
			resize(img, img, Size(screen_width * .9, new_height));
		}
		else {
			int new_width = (screen_height * aspect_ratio) * .9;
			resize(img, img, Size(new_width, screen_height * .9));
		}
	}
	cout << img.channels();
	img.convertTo(img, CV_32F);
	img = img / 255.0f;
	// Create the data vector with proper allocation
	vector<float> data(img.rows * img.cols * img.channels());
	for (int i = 0; i < img.rows * img.cols * img.channels(); i++) {
		data[i] = img.ptr<float>()[i]; // Using ptr<float>() to ensure proper conversion
	}
	//for (int i = 0;i < img.rows * img.cols * img.channels();i++) {
	//	img.data[i] = data[i];
	//}
	float BLUR3D[27]     = {1 / 27.0,1 / 27.0,1 / 27.0, 1 / 27.0,1 / 27.0,1 / 27.0, 1 / 27.0,1 / 27.0,1 / 27.0,
						   1 / 27.0,1 / 27.0,1 / 27.0, 1 / 27.0,1 / 27.0,1 / 27.0, 1 / 27.0,1 / 27.0,1 / 27.0,
						   1 / 27.0,1 / 27.0,1 / 27.0, 1 / 27.0,1 / 27.0,1 / 27.0, 1 / 27.0,1 / 27.0,1 / 27.0
	};
	float BLUR[9] = { 1 / 9.0,1 / 9.0,1 / 9.0,
						   1 / 9.0,1 / 9.0,1 / 9.0,
						   1 / 9.0,1 / 9.0,1 / 9.0
	};
	float BLUR_f[25] = { 1 / 25.0,1 / 25.0,1 / 25.0, 1 / 25.0,1 / 25.0,
					   1 / 25.0,1 / 25.0,1 / 25.0, 1 / 25.0,1 / 25.0,
					   1 / 25.0,1 / 25.0,1 / 25.0, 1 / 25.0,1 / 25.0
	};
	float HORIZ[27] = {-1.0,-2.0,-1.0, 0.0,0.0,0.0, 1.0,2.0,1.0,
					  -1.0,-2.0,-1.0, 0.0,0.0,0.0, 1.0,2.0,1.0,
					  - 1.0,-2.0,-1.0, 0.0,0.0,0.0, 1.0,2.0,1.0
	};
	dim3 vol_conv_dims = getDims(dim3(img.rows, img.cols, img.channels()), 3, 2, 1, 1);
	dim3 max_dims = getDims(dim3(img.rows, img.cols, img.channels()), 2, 0, 2, img.channels());

	clock_t host_start = clock();
	float* vol_conv_data = convolve_volume(dim3(img.rows, img.cols, img.channels()), &data[0], dim3(3, 3, 3), 1, BLUR3D, 2, 1);
	clock_t host_end = clock();
	printExecution("Host conv time", host_start, host_end);

	host_start = clock();
	float* max_data = max_pool(data.data(), dim3(img.rows,img.cols,img.channels()), 2, 0, 2);
	host_end = clock();
	printExecution("Host Max time", host_start, host_end);

	float* d_conv_data, *d_a, *h_conv_data, *d_kernel;
	h_conv_data = new float[vol_conv_dims.x * vol_conv_dims.y * vol_conv_dims.z];
	cudaMalloc(&d_conv_data, sizeof(float) * vol_conv_dims.x * vol_conv_dims.y * vol_conv_dims.z);
	cudaMalloc(&d_a, sizeof(float) * img.rows * img.cols * img.channels());
	cudaMalloc(&d_kernel, sizeof(float) * 3 * 3 * 3);
	cudaMemcpy(d_a, data.data(), sizeof(float) * img.rows * img.cols * img.channels(), cudaMemcpyHostToDevice);
	cudaMemcpy(d_kernel, BLUR3D, sizeof(float) * 3 * 3 * 3, cudaMemcpyHostToDevice);
	dim3 gridDim(vol_conv_dims.x, vol_conv_dims.y, (vol_conv_dims.z + 32 - 1) / 32);
	clock_t gpu_start = clock();
	convolve_volume << <gridDim, 32 >> > (vec3(img.rows, img.cols, img.channels()), d_a, 3, 1, d_kernel, d_conv_data, 2, 1);
	clock_t gpu_end = clock();
	printExecution("GPU conv time", gpu_start, gpu_end);
	cudaMemcpy(h_conv_data, d_conv_data, sizeof(float) * vol_conv_dims.x * vol_conv_dims.y * vol_conv_dims.z, cudaMemcpyDeviceToHost);

	float* d_max_data, * h_max_data;
	h_max_data = new float[max_dims.x * max_dims.y * max_dims.z];
	gpu_start = clock();
	cudaMalloc(&d_max_data, sizeof(float) * max_dims.x * max_dims.y * max_dims.z);
	gridDim = dim3(max_dims.x, max_dims.y, (max_dims.z + 32 - 1) / 32);
	max_pool << <gridDim, 32 >> > (vec3(img.rows, img.cols, img.channels()), d_a, 2, d_max_data, 0, 2);
	gpu_end = clock();
	printExecution("GPU Max time", gpu_start, gpu_end);
	cudaMemcpy(h_max_data, d_max_data, sizeof(float) * max_dims.x * max_dims.y * max_dims.z, cudaMemcpyDeviceToHost);
	vector<float> v_max_data(h_max_data, h_max_data+(max_dims.x * max_dims.y * max_dims.z));

	//cv::Mat image(height, width, CV_32FC(3), new_data);
	cv::Mat vol_conv_image(vol_conv_dims.x, vol_conv_dims.y, CV_32F, vol_conv_data);
	cv::Mat d_vol_conv_image(vol_conv_dims.x, vol_conv_dims.y, CV_32F, h_conv_data);
	cv::Mat max_image(max_dims.x, max_dims.y, CV_32FC(max_dims.z), max_data);
	cv::Mat d_max_image(max_dims.x, max_dims.y, CV_32FC(max_dims.z), h_max_data);
	imwrite(image_path + "mountain_pool.jpg", d_max_image);
	namedWindow("Original", WINDOW_AUTOSIZE);
	imshow("Original", img);
	waitKey(0);
	namedWindow("Convolved", WINDOW_AUTOSIZE);
	imshow("Convolved", vol_conv_image);
	waitKey(0);
	namedWindow("GPU_convolved", WINDOW_AUTOSIZE);
	imshow("GPU_convolved", d_vol_conv_image);
	waitKey(0);
	namedWindow("Max_Pool", WINDOW_AUTOSIZE);
	imshow("Max_Pool", max_image);
	waitKey(0);
	namedWindow("GPU_Max_Pool", WINDOW_AUTOSIZE);
	imshow("GPU_Max_Pool", d_max_image);
	waitKey(0);
	destroyAllWindows();
	return 0;
}