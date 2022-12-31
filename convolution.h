#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <opencv2/opencv.hpp>
#include "Operators.h"
using namespace cv;
using namespace std;

int getSize(int height, int width, int channels);

unsigned char* addImages(unsigned char* a, int aRows, int aCols, int aChannels, unsigned char* b, int bRows, int bCols, int bChannels);

void brightnessUp(unsigned char** data, int height, int width, int channels, int brightness);

void brightnessDown(unsigned char** data, int height, int width, int channels, int brightness);

void saltAndPepper(unsigned char** data, int height, int width, int channels, float prob);

void addPadding(unsigned char** data, int height, int width, int channels, int pad);

int getValue(unsigned char* data, int height, int width, int channels, int row, int col, int channel);

int convolute(unsigned char* data, int height, int width, int channels, int row, int col, int channel, float** kernel, int kernelSize);

void applyKernel(unsigned char** data, int height, int width, int channels, const double mask[][3]);

void Blur(unsigned char** data, int height, int width, int channels, float strength);

void applySepia(unsigned char** data, int height, int width, int channels);

void downSize(unsigned char** data, int height, int width, int channels, int skipRate);

void sobelEdges(unsigned char** data, int height, int width, int channels);

void prewittEdges(unsigned char** data, int height, int width, int channels);

void pooling(unsigned char** data, int height, int width, int channels, int poolSize);

unsigned char* avgPooling(unsigned char** data, int height, int width, int channels, int poolSize);

