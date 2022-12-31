#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <opencv2/opencv.hpp>
#include "Operators.h"
using namespace cv;
using namespace std;


int getSize(Mat image);

void get(Mat Image, int* width, int* height, int* channels, int* size);

Mat addImages(Mat a, Mat b);

void brightnessUp(Mat image, int brightness);

void brightnessDown(Mat image, int brightness);

void saltAndPepper(Mat image, float prob);

void addPadding(Mat image, int pad);

int getValue(Mat image, int row, int col, int channel);

int convolute(Mat image, int row, int col, int channel, float** kernel, int kernelSize);

Mat applyKernel(Mat image, const double mask[][3]);

void Blur(Mat image, float strength);

void applySepia(Mat image);

void downSize(Mat image, int skipRate);

Mat sobelEdges(Mat image);

Mat pooling(Mat image, int pools);

Mat avgPooling(Mat image, int pools);

