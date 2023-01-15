#include "convolution.h"


void avgPool(Mat* image, int poolSize);

void maxPool(Mat* image, int poolSize);

void applyKernel(Mat* image, const double mask[][3]);

void sobelEdges(Mat* image);

void prewittEdges(Mat* image);

void upscale(Mat* image, int scale);

void Blur(Mat* image, float strength);