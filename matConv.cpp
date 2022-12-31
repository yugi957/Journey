#include "matConv.h"


void avgPool(Mat* image, int poolSize) {
	int newWidth = ceil((double)(*image).cols / poolSize);
	int newHeight = ceil((double)(*image).rows / poolSize);
	unsigned char* data = avgPooling(&((*image).data), (*image).rows, (*image).cols, (*image).channels(), poolSize);
	*image = (*image)(Range(0, newHeight), Range(0, newWidth));
	(*image).data = data;
	Mat newImage;
	image->copyTo(newImage);
	*image = newImage;
}

void applyKernel(Mat* image, const double mask[][3]) {
	applyKernel(&image->data, image->rows, image->cols, image->channels(), TEST);

}

void sobelEdges(Mat* image) {
	sobelEdges(&((*image).data), image->rows, image->cols, image->channels());
}

void prewittEdges(Mat* image) {
	prewittEdges(&((*image).data), image->rows, image->cols, image->channels());
}