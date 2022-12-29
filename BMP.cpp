#include "BMP.h"
#include <string>
#include <stdio.h>
#include <errno.h>

using namespace std;


BMP::BMP()
{
    //ctor
}

BMP::BMP(Image a, unsigned char head[54], unsigned char ct[1024], int bd, char* str) {
    for (int i = 0;i < 54;i++) {
        this->header[i] = head[i];
    }
    for (int i = 0;i < 1024;i++) {
        this->colorTable[i] = ct[i];
    }
    this->bitDepth = bd;
    this->width = a.width;
    this->height = a.height;
    this->size = a.size;
    this->data = a.data;
    this->name = str;
    this->color = a.color;
    this->channels = a.channels;

}

void BMP::printHeader() {
    for (int i = 0;i < 54;i++) {
        printf("byte %d: %d\n", i, *(int*)&header[i]);
    }
}

void BMP::readFile(char* filename, colorType cType) {
    FILE* streamIn = fopen(filename, "rb");
    name = getString(filename);
    color = cType;

    if (streamIn == (FILE*)0) {
        printf("Unable to open file\n");
    }

    for (int i = 0;i < 54;i++) {
        header[i] = getc(streamIn);
    }

    width = *(int*)&header[18];
    height = *(int*)&header[22];
    bitDepth = *(int*)&header[28];
    if (cType == GRAYSCALE) {
        size = width * height;
        channels = 1;
    }
    else if (cType == RGB) {
        size = width * height * 3;
        channels = 3;
    }

    if (bitDepth <= 8) {
        fread(colorTable, sizeof(unsigned char), 1024, streamIn);
    }

    data = (unsigned char*)malloc(sizeof(unsigned char) * size);

    fread(data, sizeof(unsigned char), size, streamIn);

    fclose(streamIn);
    printf("Success !\n");
}

void BMP::writeFile(char* filename) {
    errno = 0;

    FILE* streamOut = fopen(filename, "wb");
    printf("ERROR: %d\n", errno);

    fwrite(header, sizeof(unsigned char), 54, streamOut);
    if (bitDepth <= 8) {
        fwrite(colorTable, sizeof(unsigned char), 1024, streamOut);
    }

    for (int i = 0;i < size;i++) {
        fwrite(data+i, sizeof(unsigned char), 1, streamOut);
    }
    fwrite(data, sizeof(unsigned char), size, streamOut);
}

void  BMP::compareFile(BMP a, BMP b) {
    bool errCheck = true;
    if (a.size != b.size) {
        printf("SIZES NOT THE SAME\n");
        return;
    }
    for (int i = 0;i < BMP_HEADER_SIZE;i++) {
        if (a.header[i] != b.header[i]) {
            printf("HEADER DIFF - %d :: %d\n", a.header[i], b.header[i]);
            errCheck = false;
        }
    }
    for (int i = 0;i < BMP_COLOR_TABLE_SIZE;i++) {
        if (a.colorTable[i] != b.colorTable[i]) {
            printf("COLORTABLE DIFF - %d :: %d\n", a.colorTable[i], b.colorTable[i]);
            errCheck = false;
        }
    }
    for (int i = 0;i < a.size;i++) {
        if (a.data[i] != b.data[i]) {
            printf("DATA DIFF - pixel %d - %d :: %d\n", i, a.data[i], b.data[i]);
            errCheck = false;
        }
    }
    if (errCheck) printf("Files are the same!\n");
}


BMP BMP::addImages(BMP a, BMP b) {
    if (a.width != b.width || a.height != b.height || a.color != b.color) {
        printf("IMAGES NOT ADDABLE");
        return a;
    }

    unsigned char* newData = new unsigned char[a.width * b.height * a.channels];
    
    for (int i = 0;i < a.size;i++) {
        int pixel = a.data[i] + b.data[i];
        newData[i] = (pixel > 255) ? 255 : pixel;
    }

    //char* temp = new char[(sizeof(a.name) / sizeof(char)) + (sizeof(b.name) / sizeof(char)) + 1];
    //strcat(temp, a.name);
    //strcat(temp, "&");
    //strcat(temp, b.name);

    BMP res = BMP(a, a.header, a.colorTable, a.bitDepth, "addedImage");
    res.data = newData;
    return res;


}

BMP::~BMP()
{
    //dtor
}
