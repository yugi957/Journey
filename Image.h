#ifndef IMAGE_H
#define IMAGE_H

#include <string>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

enum colorType {
    GRAYSCALE,
    RGB
};
#include "Operators.h"
#define MAX_BRIGHTNESS 255;

class Image
{
public:
    Image();
    Image(Mat image);
    virtual ~Image();

    int width;
    int height;
    int size;
    int concreteWidth;
    int concreteHeight;
    int concreteSize;
    unsigned char* data;
    char* name;
    float hist[256];
    colorType color;
    int channels;
    void writeData(char* filename);
    char* getString(char* str);
    void brightnessUp(int brightness);
    void brightnessDown(int brightness);
    void saltAndPepper(float prob);
    void addPadding(int pad);

    void computeHistogram();
    void computeHistogram(char* filename);
    void equalizeHistogram();
    void writeHist(char* filename);

    int getValue(int row, int col, int channel);
    int convolute(int row, int col, int channel, float** kernel, int kernelSize);
    void applyKernel(const double kernel[][3]);
    void Blur(float strength);
    void applySepia();
    void parallelConv(const double kernel[][3]);
    void parallelBlur(double strength);

    void downSize(int skipRate);



protected:

private:
};

#endif // IMAGE_H
