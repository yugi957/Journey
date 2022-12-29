#include "Image.h"
#include "CUDAFunctions.cuh"

void Image::parallelConv(const double kernel[][3]) {
    unsigned char* convData = (unsigned char*)malloc(sizeof(unsigned char) * size);
    double* kern = new double [9];

    for (int i = 0;i < 3;i++) {
        for (int j = 0;j < 3;j++) {
            kern[i * 3 + j] = kernel[i][j];
        }
    }

    convolution(data, convData, kern, height, width, channels, size, 3);

    data = convData;
}

void Image::parallelBlur(double strength) {
    unsigned char* convData = (unsigned char*)malloc(sizeof(unsigned char) * size);
    double* kern = new double[strength * strength];

    for (int i = 0;i < strength;i++) {
        for (int j = 0;j < strength;j++) {
            kern[i * (int)strength + j] = 1 / (strength * strength);
        }
    }

    convolution(data, convData, kern, height, width, channels, size, strength);

    data = convData;
}