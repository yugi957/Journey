#include "convolution.h"

int getSize(int height, int width, int channels) {
    return height * width * channels;
}

unsigned char* addImages(unsigned char* a, int aRows, int aCols, int aChannels, unsigned char* b, int bRows, int bCols, int bChannels) {
    if (aCols != bCols || aRows != bRows || aChannels != bChannels) {
        printf("IMAGES NOT ADDABLE");
        return a;
    }
    int size = getSize(aRows, aCols, aChannels);
    unsigned char* newData = (unsigned char*)malloc(sizeof(unsigned char) * size);

    for (int i = 0;i < size;i++) {
        int pixel = a[i] + b[i];
        newData[i] = (pixel > 255) ? 255 : pixel;
    }

    return newData;
}

void brightnessUp(unsigned char** data, int height, int width, int channels, int brightness){
    //This uses truncation to ensure it doesn't go above 255
    int size = getSize(height, width, channels);

    for (int i = 0;i < size;i++) {
        int temp = (*data)[i] + brightness;
        (*data)[i] = (temp > 255) ? 255 : temp;
    }
}

void brightnessDown(unsigned char** data, int height, int width, int channels, int brightness) {
    //This uses truncation to ensure it doesn't go below 0
    int size = getSize(height, width, channels);

    for (int i = 0;i < size;i++) {
        int temp = (*data)[i] - brightness;
        (*data)[i] = (temp < 0) ? 0 : temp;
    }
}

void saltAndPepper(unsigned char** data, int height, int width, int channels, float prob) {
    int size = getSize(height, width, channels);
    int x, y, p1, p2, p;
    p = (int)(prob * 32768 / 2);
    p1 = p + 16384;
    p2 = 16384 - p;

    for (int r = 0;r < height;r++)
        for (int c = 0;c < width;c++)
        {
            p = rand();
            if (p >= 16384 && p < p1)
                for (int chan = 0;chan < channels;chan++) (*data)[r * width * channels + c * channels + chan] = 0;
            if (p >= p2 && p < 16384)
                for (int chan = 0;chan < channels;chan++) (*data)[r * width * channels + c * channels + chan] = 255;
        }
}

void addPadding(unsigned char** data, int height, int width, int channels, int pad) {
    int size = getSize(height, width, channels);
    int tempWidth = width + pad * 2;
    int tempHeight = height + pad * 2;
    unsigned char* newData = (unsigned char*)malloc(sizeof(unsigned char) * (tempWidth * tempHeight) * (channels));

    for (int r = 0;r < tempHeight;r++) {
        for (int c = 0;c < tempWidth;c++) {
            for (int chan = 0;chan < channels;chan++) {
                int newPixel = r * tempWidth * channels + c * channels + chan;
                int pixel = (r - 1) * width * channels + (c - 1) * channels + chan;
                if (r == 0 || c == 0 || r == tempHeight - 1 || c == tempWidth - 1) newData[newPixel] = 0;
                else newData[newPixel] = (*data)[pixel];
            }
        }
    }
    size = tempHeight * tempWidth;
    (*data) = newData;
}

int getValue(unsigned char* data, int height, int width, int channels, int row, int col, int channel) {
    if (row < height && row >= 0 && col < width && col >= 0) {
        return data[(row * width * channels) + (col * channels) + channel];
    }
    return 0;
}

int convolute(unsigned char* data, int height, int width, int channels, int row, int col, int channel, float** kernel, int kernelSize) {
    float sum = 0;
    int end = (int)((kernelSize / 2.0) + .5);
    int start = end - kernelSize;
    for (int i = start;i < end;i++) {
        for (int j = start;j < end;j++) {
            sum += getValue(data, height, width, channels, row + i, col + j, channel) * kernel[i + (kernelSize / 2)][j + (kernelSize / 2)];
        }
    }
    if (sum > 255) return 255;
    if (sum < 0) return 0;
    return (int)sum;
}

int poolMax(unsigned char* data, int height, int width, int channels, int row, int col, int channel, int poolSize) {
    float sum = 0;
    int end = (int)((poolSize / 2.0) + .5);
    int start = end - poolSize;
    int max = 0;
    int pixel = 0;
    for (int i = start;i < end;i++) {
        for (int j = start;j < end;j++) {
            pixel = getValue(data, height, width, channels, row + i, col + j, channel);
            if (pixel > max) max = pixel;
        }
    }
    return max;
}

int poolAvg(unsigned char* data, int height, int width, int channels, int row, int col, int channel, int poolSize) {
    float sum = 0;
    int end = (int)((poolSize / 2.0) + .5);
    int start = end - poolSize;
    double avg = 0;
    for (int i = start;i < end;i++) {
        for (int j = start;j < end;j++) {
            avg += getValue(data, height, width, channels, row + i, col + j, channel);
        }
    }
    return (int)(avg / (poolSize * poolSize));
}

void applyKernel(unsigned char** data, int height, int width, int channels, const double mask[][3]) {
    int size = getSize(height, width, channels);
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
    printf("size: %d\n", sizeof(newData));
    int pixel = 0;
    int count = 0;
    int row = 0;
    int col = 0;
    for (int r = 0;r < height;r++) {
        for (int c = 0;c < width;c++) {
            for (int chan = 0;chan < channels;chan++) {
                pixel = (r * width * channels) + (c * channels) + chan;
                row = r * width;
                col = c;
                //printf("convoluting pixel %d...\n", pixel);
                newData[pixel] = convolute((*data), height, width, channels, r, c, chan, kernel, 3);
                count++;
            }
        }
    }
    (*data) = newData;
}

void Blur(unsigned char** data, int height, int width, int channels, float strength) {
    int size = getSize(height, width, channels);
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
                newData[pixel] = convolute((*data), height, width, channels, r, c, chan, kernel, 3);
            }
        }
    }

    (*data) = newData;
}

void applySepia(unsigned char** data, int height, int width, int channels) {
    int size = getSize(height, width, channels);
    float sepia[3][3] = { {.393,.769,.189}, {.349,.686,.168}, {.272,.534,.131} };
    int r, g, b;
    for (int i = 0;i < size;i += 3) {
        r = (*data)[i] * sepia[0][0] + (*data)[i + 1] * sepia[0][1] + (*data)[i + 2] * sepia[0][2];
        g = (*data)[i] * sepia[1][0] + (*data)[i + 1] * sepia[1][1] + (*data)[i + 2] * sepia[1][2];
        b = (*data)[i] * sepia[2][0] + (*data)[i + 1] * sepia[2][1] + (*data)[i + 2] * sepia[2][2];
        (*data)[i] = (b < 255) ? b : 255;
        (*data)[i + 1] = (g < 255) ? g : 255;
        (*data)[i + 2] = (r < 255) ? r : 255;
    }
}

void downSize(unsigned char** data, int height, int width, int channels, int skipRate) {
    int size = getSize(height, width, channels);
    unsigned char* newData = (unsigned char*)malloc(sizeof(unsigned char) * size);

    //just for better viewing
    for (int r = 0;r < height;r++) {
        for (int c = 0;c < width;c++) {
            for (int chan = 0;chan < channels;chan++) {
                newData[(r * width * channels) + (c * channels) + chan] = 0;
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
                        newData[newR * width * channels + newC * channels + chan] = (*data)[r * width * channels + c * channels + chan];
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
    (*data) = newData;
}

void sobelEdges(unsigned char** data, int height, int width, int channels) {
    int size = getSize(height, width, channels);
    unsigned char* horiz = *data;
    applyKernel(&horiz, height, width, channels, SOBEL_HORIZ);
    unsigned char* vert = *data;
    applyKernel(&vert, height, width, channels, SOBEL_VERT);
    (*data) = addImages(horiz, height, width, channels, vert, height, width, channels);
}

void prewittEdges(unsigned char** data, int height, int width, int channels) {
    int size = getSize(height, width, channels);
    unsigned char* horiz = *data;
    applyKernel(&horiz, height, width, channels, PREWITT_HORIZ);
    unsigned char* vert = *data;
    applyKernel(&vert, height, width, channels, PREWITT_VERT);
    (*data) = addImages(horiz, height, width, channels, vert, height, width, channels);
}

void pooling(unsigned char** data, int height, int width, int channels, int poolSize) {
    int size = getSize(height, width, channels);

    int newWidth = ceil((double)width / poolSize);
    int newHeight = ceil((double)height / poolSize);
    int newSize = width * newHeight * channels;
    unsigned char* newData = (unsigned char*)malloc(sizeof(unsigned char) * newSize);

    int rStart = poolSize / 2;
    int cStart = poolSize / 2;
    printf("POOLING %dx%d IMAGE TO %dx%d IMAGE\n", height, width, newHeight, newWidth);
    int newC, newR;
    newC = newR = 0;
    for (int r = rStart;r < rStart * height;r += poolSize) {
        for (int c = cStart;c < cStart * width;c += poolSize) {
            for (int chan = 0;chan < channels;chan++) {
                newData[(newR * width * channels) + (newC * channels) + chan] = poolMax((*data), height, width, channels, r, c, chan, poolSize);
            }
            newC++;
        }
        newR++;
        newC = 0;
    }
    (*data) = newData;
}

unsigned char* avgPooling(unsigned char** data, int height, int width, int channels, int poolSize) {
    int size = getSize(height, width, channels);

    int newWidth = ceil((double)width / poolSize);
    int newHeight = ceil((double)height / poolSize);
    int newSize = width * newHeight * channels;
    unsigned char* newData = (unsigned char*)malloc(sizeof(unsigned char) * newSize);

    int rStart = poolSize / 2;
    int cStart = poolSize / 2;
    printf("POOLING %dx%d IMAGE TO %dx%d IMAGE\n", height, width, newHeight, newWidth);
    int newC, newR;
    newC = newR = 0;
    for (int r = rStart;r < height;r += poolSize) {
        for (int c = cStart;c < width;c += poolSize) {
            for (int chan = 0;chan < channels;chan++) {
                newData[(newR * width * channels) + (newC * channels) + chan] = poolAvg((*data), height, width, channels, r, c, chan, poolSize);
            }
            newC++;
        }
        newR++;
        newC = 0;
    }
    return newData;
}