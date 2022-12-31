#include "CVImage.h"

int getSize(Mat image) {
    return image.rows * image.cols * image.channels();
}

void get(Mat Image, int* width, int* height, int* channels, int* size) {
    *width = Image.cols;
    *height = Image.rows;
    *channels = Image.channels();
    *size = (*width) * (*height) * (*channels);

}

Mat addImages(Mat a, Mat b) {
    if (a.cols != b.cols || a.rows != b.rows || a.channels() != b.channels()) {
        printf("IMAGES NOT ADDABLE");
        return a;
    }
    int width, height, channels, size;
    get(a, &width, &height, &channels, &size);
    unsigned char* newData = (unsigned char*)malloc(sizeof(unsigned char) * size);

    for (int i = 0;i < size;i++) {
        int pixel = a.data[i] + b.data[i];
        newData[i] = (pixel > 255) ? 255 : pixel;
    }

    a.data = newData;
    return a;
}

void brightnessUp(Mat image, int brightness) {
    //This uses truncation to ensure it doesn't go above 255
    int width, height, channels, size;
    get(image, &width, &height, &channels, &size);

    for (int i = 0;i < size;i++) {
        int temp = image.data[i] + brightness;
        image.data[i] = (temp > 255) ? 255 : temp;
    }
}

void brightnessDown(Mat image, int brightness) {
    //This uses truncation to ensure it doesn't go below 0
    int width, height, channels, size;
    get(image, &width, &height, &channels, &size);

    for (int i = 0;i < size;i++) {
        int temp = image.data[i] - brightness;
        image.data[i] = (temp < 0) ? 0 : temp;
    }
}

void saltAndPepper(Mat image, float prob) {
    int width, height, channels, size;
    get(image, &width, &height, &channels, &size);
    int x, y, p1, p2, p;
    p = (int)(prob * 32768 / 2);
    p1 = p + 16384;
    p2 = 16384 - p;

    for (int r = 0;r < height;r++)
        for (int c = 0;c < width;c++)
        {
            p = rand();
            if (p >= 16384 && p < p1)
                for (int chan = 0;chan < channels;chan++) image.data[r * width * channels + c * channels + chan] = 0;
            if (p >= p2 && p < 16384)
                for (int chan = 0;chan < channels;chan++) image.data[r * width * channels + c * channels + chan] = 255;
        }
}

void addPadding(Mat image, int pad) {
    int width, height, channels, size;
    get(image, &width, &height, &channels, &size);
    int tempWidth = width + pad * 2;
    int tempHeight = height + pad * 2;
    unsigned char* newData = (unsigned char*)malloc(sizeof(unsigned char) * (tempWidth * tempHeight) * (channels));

    for (int r = 0;r < tempHeight;r++) {
        for (int c = 0;c < tempWidth;c++) {
            for (int chan = 0;chan < channels;chan++) {
                int newPixel = r * tempWidth * channels + c * channels + chan;
                int pixel = (r - 1) * width * channels + (c - 1) * channels + chan;
                if (r == 0 || c == 0 || r == tempHeight - 1 || c == tempWidth - 1) newData[newPixel] = 0;
                else newData[newPixel] = image.data[pixel];
            }
        }
    }
    size = tempHeight * tempWidth;
    image.data = newData;
}

int getValue(Mat image, int row, int col, int channel) {
    int width, height, channels, size;
    get(image, &width, &height, &channels, &size);
    if (row < height && row >= 0 && col < width && col >= 0) {
        return image.data[(row * width * channels) + (col * channels) + channel];
    }
    return 0;
}

int convolute(Mat image, int row, int col, int channel, float** kernel, int kernelSize) {
    float sum = 0;
    int end = (int)((kernelSize / 2.0) + .5);
    int start = end - kernelSize;
    for (int i = start;i < end;i++) {
        for (int j = start;j < end;j++) {
            sum += getValue(image, row + i, col + j, channel) * kernel[i + (kernelSize / 2)][j + (kernelSize / 2)];
        }
    }
    if (sum > 255) return 255;
    if (sum < 0) return 0;
    return (int)sum;
}

int poolMax(Mat image, int row, int col, int channel, int poolSize) {
    float sum = 0;
    int end = (int)((poolSize / 2.0) + .5);
    int start = end - poolSize;
    int max = 0;
    int pixel = 0;
    for (int i = start;i < end;i++) {
        for (int j = start;j < end;j++) {
            pixel = getValue(image, row + i, col + j, channel);
            if (pixel > max) max = pixel;
        }
    }
    return max;
}

int poolAvg(Mat image, int row, int col, int channel, int poolSize) {
    int width, height, channels, size;
    get(image, &width, &height, &channels, &size);
    float sum = 0;
    int end = (int)((poolSize / 2.0) + .5);
    int start = end - poolSize;
    double avg = 0;
    for (int i = start;i < end;i++) {
        for (int j = start;j < end;j++) {
            avg += getValue(image, row + i, col + j, channel);
        }
    }
    return (int)(avg / (poolSize*poolSize));
}

Mat applyKernel(Mat image, const double mask[][3]) {
    int width, height, channels, size;
    get(image, &width, &height, &channels, &size);
    float** kernel = new float* [3];
    for (int i = 0; i < 3; i++) {
        kernel[i] = new float[3];
    }

    for (int i = 0;i < 3;i++) {
        for (int j = 0;j < 3;j++) {
            kernel[i][j] = mask[i][j];
        }
    }
    Mat newImage = image.clone();
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
                //newData[pixel] = convolute(image, r, c, chan, kernel, 3);
                newImage.data[pixel] = convolute(image, r, c, chan, kernel, 3);
                count++;
            }
        }
        namedWindow("Display frame", WINDOW_AUTOSIZE);
        imshow("Display frame", newImage);
        waitKey(1);
    }
    //image.data = newData;
    for (int i = 0;i < size;i++) {
        image.data[i] = newData[i];
    }
    //imshow("Display frame", image);
    //waitKey(0);
    return newImage;
}

void Blur(Mat image, float strength) {
    int width, height, channels, size;
    get(image, &width, &height, &channels, &size);
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
                newData[pixel] = convolute(image, r, c, chan, kernel, (int)strength);
            }
        }
    }

    image.data = newData;
}

void applySepia(Mat image) {
    int width, height, channels, size;
    get(image, &width, &height, &channels, &size);
    float sepia[3][3] = { {.393,.769,.189}, {.349,.686,.168}, {.272,.534,.131} };
    int r, g, b;
    for (int i = 0;i < size;i += 3) {
        r = image.data[i] * sepia[0][0] + image.data[i + 1] * sepia[0][1] + image.data[i + 2] * sepia[0][2];
        g = image.data[i] * sepia[1][0] + image.data[i + 1] * sepia[1][1] + image.data[i + 2] * sepia[1][2];
        b = image.data[i] * sepia[2][0] + image.data[i + 1] * sepia[2][1] + image.data[i + 2] * sepia[2][2];
        image.data[i] = (b < 255) ? b : 255;
        image.data[i + 1] = (g < 255) ? g : 255;
        image.data[i + 2] = (r < 255) ? r : 255;
    }
}

void downSize(Mat image, int skipRate) {
    int width, height, channels, size;
    get(image, &width, &height, &channels, &size);
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
                        newData[newR * width * channels + newC * channels + chan] = image.data[r * width * channels + c * channels + chan];
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
    image.data = newData;
}

Mat sobelEdges(Mat image) {
    Mat image2 = image.clone();
    image = applyKernel(image, SOBEL_HORIZ);
    image2 = applyKernel(image2, SOBEL_VERT);
    image = addImages(image, image2);
    return image;
}

Mat pooling(Mat image, int poolSize) {
    int width, height, channels, size;
    get(image, &width, &height, &channels, &size);

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
                newData[(newR * width * channels) + (newC * channels) + chan] = poolMax(image, r, c, chan, poolSize);
            }
            newC++;
        }
        newR++;
        newC = 0;
    }
    //printf("%dx%d\n", image.rows, image.cols);
    image = image(Range(0, newHeight), Range(0, newWidth));
    //printf("%dx%d\n", image.rows, image.cols);
    image.data = newData;
    return image;
}

Mat avgPooling(Mat image, int poolSize) {
    int width, height, channels, size;
    get(image, &width, &height, &channels, &size);

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
                newData[(newR * width * channels) + (newC * channels) + chan] = poolAvg(image, r, c, chan, poolSize);
            }
            newC++;
        }
        newR++;
        newC = 0;
        //namedWindow("Display frame", WINDOW_AUTOSIZE);
        //imshow("Display frame", image);
        //waitKey(1);
    }
    
    image = image(Range(0, newHeight), Range(0, newWidth));
    image.data = newData;
    return image;
}