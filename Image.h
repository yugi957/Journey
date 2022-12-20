#ifndef IMAGE_H
#define IMAGE_H

#include <string>
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
    virtual ~Image();

    int width;
    int height;
    int size;
    unsigned char* data;
    char* name;
    float hist[256];
    colorType color;
    int channels;
    void writeData(char* filename);
    char* getString(char* str);
    void brightnessUp(int brightness);
    void brightnessDown(int brightness);
    void Image::saltAndPepper(float prob);

    void computeHistogram();
    void computeHistogram(char* filename);
    void equalizeHistogram();
    void writeHist(char* filename);

    int getValue(int row, int col, int channel);
    int convolute(int row, int col, int channel, float** kernel, int kernelSize);
    void applyKernel(const float kernel[][3]);
    void Convolve2D(const float mask[][3]);
    void Blur(float strength);
    void applySepia();



protected:

private:
};

#endif // IMAGE_H
