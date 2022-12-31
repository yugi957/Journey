#include "Image.h"
#include <stdio.h>
#include <stdlib.h>



Image::Image()
{
    //ctor
}

Image::Image(Mat image) {
    width = image.cols;
    height = image.rows;
    channels = image.channels();
    size = width * height * channels;
    data = image.data;


}

void Image::writeData(char* filename) {
    FILE* dataFile;
    dataFile = fopen(filename, "w");
    fprintf(dataFile, "Size: %d\n", size);
    for (int i = 0;i < concreteSize;i++) {
        fprintf(dataFile, "pixel %d: %d\n", i, data[i]);
    }
    fclose(dataFile);
}

char * Image::getString(char* str) {
    int ind = 0;
    int p = 0;
    for (int i = 0;i < 1000;i++) {
        if (str[i] == '/' || str[i] == '\\') ind = i+1;
        if (str[i] == '.') {
            p = i;
            break;
        }
    }
    int l = p - ind;
    char* res = new char[l];
    for (int i = 0;i < l;i++) {
        res[i] = str[i + ind];
    }
    res[l] = '\0';

    return res;
}

void Image::brightnessUp(int brightness) {
    //This uses truncation to ensure it doesn't go above 255

    for (int i = 0;i < size;i++) {
        int temp = data[i] + brightness;
        data[i] = (temp > 255) ? 255 : temp;
    }
}

void Image::brightnessDown(int brightness) {
    //This uses truncation to ensure it doesn't go below 0

    for (int i = 0;i < size;i++) {
        int temp = data[i] - brightness;
        data[i] = (temp < 0) ? 0 : temp;
    }
}

void Image::saltAndPepper(float prob){
    int x, y, p1, p2, p;
    p = (int)(prob * 32768 / 2);
    p1 = p + 16384;
    p2 = 16384 - p;

    for (int r = 0;r < height;r++)
        for (int c = 0;c < width;c++)
        {
            p = rand();
            if (p >= 16384 && p < p1)
                for(int chan = 0;chan < channels;chan++) data[r * width * channels + c * channels + chan] = 0;
            if (p >= p2 && p < 16384)
                for (int chan = 0;chan < channels;chan++) data[r * width * channels + c * channels + chan] = 255;
        }
}

void Image::addPadding(int pad) {
    int tempWidth = width + pad * 2;
    int tempHeight = height + pad * 2;
    unsigned char* newData = (unsigned char*)malloc(sizeof(unsigned char) * (tempWidth * tempHeight) * (channels));

    for (int r = 0;r < tempHeight;r++) {
        for (int c = 0;c < tempWidth;c++) {
            for (int chan = 0;chan < channels;chan++) {
                int newPixel = r * tempWidth * channels + c * channels + chan;
                int pixel = (r-1) * width * channels + (c-1) * channels + chan;
                if (r == 0 || c == 0 || r == tempHeight -1 || c == tempWidth -1) newData[newPixel] = 0;
                else newData[newPixel] = data[pixel];
            }
        }
    }
    size = tempHeight * tempWidth;
    data = newData;
}

void Image::computeHistogram() {
    char* temp = new char[(sizeof(name) / sizeof(name[0])) + 16];
    temp[0] = '\0';
    strcat(temp, "images/");
    strcat(temp, name);
    strcat(temp, "_hist.txt");

    long int ihist[256], sum;
    for (int i = 0;i < 256;i++) {
        ihist[i] = 0;
    }

    int pixel = 0;
    for (int i = 0;i < size;i++) {
        pixel = data[i];
        ihist[pixel] += 1;
    }
    for (int i = 0;i < 256;i++) {
        hist[i] = ihist[i] / (float)size;
    }
    writeHist(temp);
}

void Image::computeHistogram(char* filename) {
    long int ihist[256], sum;
    for (int i = 0;i < 256;i++) {
        ihist[i] = 0;
    }

    int pixel = 0;
    for (int i = 0;i < size;i++) {
        pixel = data[i];
        ihist[pixel] += 1;
    }
    for (int i = 0;i < 256;i++) {
        hist[i] = ihist[i] / (float)size;
    }
    writeHist(filename);
}

void Image::equalizeHistogram() {
    FILE* histFile;
    char* temp = new char[(sizeof(name) / sizeof(name[0])) + 18];
    temp[0] = '\0';
    strcat(temp, "images/");
    strcat(temp, name);
    strcat(temp, "_eqHist.txt");

    histFile = fopen(temp, "w");

    float sum;
    int eqHist[256];

    for (int i = 0;i < 256;i++) {
        sum = 0.0;
        for (int j = 0;j <= i;j++) {
            sum += hist[j];
        }
        eqHist[i] = (int)(255 * sum + 0.5);
    }

    for (int i = 0;i < size;i++) {
        data[i] = eqHist[data[i]];
    }
    computeHistogram(temp);
}

void Image::writeHist(char* filename) {
    FILE* histFile;
    histFile = fopen(filename, "w");
    for (auto pixel : hist) {
        fprintf(histFile, "%f\n", pixel);
    }
    fclose(histFile);
}


int Image::getValue(int row, int col, int channel) {
    if (row < height && row >= 0 && col < width && col >= 0) {
        return data[(row * width * channels) + (col * channels) + channel];
    }
    return 0;
}

int Image::convolute(int row, int col, int channel, float** kernel, int kernelSize) {
    float sum = 0;
    int end = (int)((kernelSize / 2.0) + .5);
    int start = end - kernelSize;
    for (int i = start;i < end;i++) {
        for (int j = start;j < end;j++) {
            sum += getValue(row + i, col + j, channel) * kernel[i + (kernelSize / 2)][j + (kernelSize / 2)];
        }
    }
    if (sum > 255) return 255;
    if (sum < 0) return 0;
    return (int)sum;
}

void Image::applyKernel(const double mask[][3]) {
    float** kernel = new float* [3];
    for (int i = 0; i < 3; i++) {
        kernel[i] = new float[3];
    }

    for (int i = 0;i < 3;i++) {
        for (int j = 0;j < 3;j++) {
            kernel[i][j] = mask[i][j];
        }
    }
    unsigned char* newData = (unsigned char*)malloc(sizeof(unsigned char) * size);
    int pixel = 0;
    int count = 0;
    for (int r = 0;r < height;r++) {
        for (int c = 0;c < width;c++) {
            for (int chan = 0;chan < channels;chan++) {
                pixel = (r * width * channels) + (c * channels) + chan;
                //printf("convoluting pixel %d...\n", pixel);
                newData[pixel] = convolute(r, c, chan, kernel, 3);
                count++;
            }
        }
    }

    data = newData;
}

void Image::Blur(float strength) {
    float** kernel = new float* [strength];
    for (int i = 0; i < strength; i++) {
        kernel[i] = new float[strength];
    }

    for (int i = 0;i < strength;i++) {
        for (int j = 0;j < strength;j++) {
            kernel[i][j] = 1 / (strength * strength);
        }
    }

    unsigned char* newData = (unsigned char*)malloc(sizeof(unsigned char) * size);

    int pixel = 0;
    int count = 0;
    for (int r = 0;r < height;r++) {
        for (int c = 0;c < width;c++) {
            for (int chan = 0;chan < channels;chan++) {
                pixel = (r * width * channels) + (c * channels) + chan;
                //printf("convoluting pixel %d...\n", pixel);
                newData[pixel] = convolute(r, c, chan, kernel, (int)strength);
            }
        }
    }

    data = newData;
}

void Image::applySepia() {
    float sepia[3][3] = {{.393,.769,.189}, {.349,.686,.168}, {.272,.534,.131}};
    int r, g, b;
    for (int i = 0;i < size;i+=3) {
        r = data[i] * sepia[0][0] + data[i + 1] * sepia[0][1] + data[i + 2] * sepia[0][2];
        g = data[i] * sepia[1][0] + data[i + 1] * sepia[1][1] + data[i + 2] * sepia[1][2];
        b = data[i] * sepia[2][0] + data[i + 1] * sepia[2][1] + data[i + 2] * sepia[2][2];
        data[i] = (b < 255) ? b : 255;
        data[i + 1] = (g < 255) ? g : 255;
        data[i + 2] = (r < 255) ? r : 255;
    }
}

void Image::downSize(int skipRate) {


    unsigned char* newData = (unsigned char*)malloc(sizeof(unsigned char) * concreteSize);

    //just for better viewing
    for (int r = 0;r < concreteHeight;r++) {
        for (int c = 0;c < concreteWidth;c++) {
            for (int chan = 0;chan < channels;chan++) {
                newData[(r * concreteWidth * channels) + (c * channels) + chan] = 0;
            }
        }
    }

    int newR, newC;
    int lastC = 0;
    newR = newC = 0;
    int r = 0;
    int c = 0;
    for (r = 0;r < height;r++) {
        if (r % skipRate != 0) {
            for (c = 0;c < width;c++) {
                if (c % skipRate != 0) {
                    for (int chan = 0;chan < channels;chan++) {
                        newData[newR * concreteWidth * channels + newC * channels + chan] = data[r * concreteWidth * channels + c * channels + chan];
                    }
                    newC++;
                }
            }
            newR++;
        }
        if (newC > lastC) lastC = newC;
        newC = 0;
    }

    height = newR;
    width = lastC;
    size = height * width * channels;
    data = newData;
}

Image::~Image()
{
    //dtor
}
