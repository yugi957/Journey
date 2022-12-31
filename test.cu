//#include "CUDAFunctions.cuh";
#include "Image.h"
#include "image.cuh";
#include "BMP.h"
#include "MLP.h"
#include "CVImage.h"
#include <opencv2/opencv.hpp>
#include "convolution.h"
#include "matConv.h"
using namespace cv;


int main() {
	//namedWindow("Display frame", WINDOW_AUTOSIZE);
	Mat image = imread("train/1.jpg", IMREAD_GRAYSCALE);
	imshow("Display frame", image);
	waitKey(0);
	avgPool(&image, 3);
	imshow("Display frame", image);
	waitKey(0);
	Mat image2 = image.clone();
	sobelEdges(&image);
	imshow("Display frame", image);
	waitKey(0);
	prewittEdges(&image2);
	imshow("Display frame", image2);
	waitKey(0);
	//imwrite("jpgs/idk_small.jpg", image);
	//applyKernel(image2, TEST);
	//applyKernel(&image2.data, image2.rows, image2.cols, image2.channels(), TEST);
	//imshow("Display frame", image2);
	//waitKey(0);






}