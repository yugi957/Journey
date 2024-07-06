#include "../general.h"
#include "../CUDA/Headers/general.cuh"
#include <opencv2/opencv.hpp>
#include "Headers/ImageProcessing.cuh"
#include <fstream>
using namespace cv;

int main() {

	string image_path = "C:/Atul/cuDNN/images/";
	string full_path = image_path + "idk.bmp";

	ifstream file(full_path);
	if (!file) {
		cout << "File does not exist: " << full_path << endl;
		//return -1;
	}
	file.close();
	Mat img = imread(image_path + "idk.bmp");
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
	dim3 conv_dims = getDims(dim3(img.rows, img.cols, img.channels()), 5, 2, 1, img.channels());
	dim3 vol_conv_dims = getDims(dim3(img.rows, img.cols, img.channels()), 3, 2, 1, 1);
	dim3 max_dims = getDims(vol_conv_dims, 2, 0, 2, vol_conv_dims.z);

	float* conv_data = convolve(dim3(img.rows, img.cols, img.channels()), &data[0], dim3(5, 5, 1), 3, BLUR_f, 2, 1);
	float* vol_conv_data = convolve_volume(dim3(img.rows, img.cols, img.channels()), &data[0], dim3(3, 3, 3), 1, HORIZ, 2, 1);
	float* max_data = max_pool(vol_conv_data, vol_conv_dims, 2, 0, 2);

	//cv::Mat image(height, width, CV_32FC(3), new_data);
	cv::Mat vol_conv_image(vol_conv_dims.x, vol_conv_dims.y, CV_32F, vol_conv_data);
	cv::Mat conv_image(conv_dims.x, conv_dims.y, CV_32FC(3), conv_data);
	cv::Mat max_image(max_dims.x, max_dims.y, CV_32FC(max_dims.z), max_data);
	namedWindow("Original", WINDOW_AUTOSIZE);
	imshow("Original", img);
	waitKey(0);

	namedWindow("Convolved", WINDOW_AUTOSIZE);
	imshow("Convolved", vol_conv_image);
	waitKey(0);
	namedWindow("Max_Pool", WINDOW_AUTOSIZE);
	imshow("Max_Pool", max_image);
	waitKey(0);

	destroyAllWindows();
	return 0;
}