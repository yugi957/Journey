#ifndef BMP_H
#define BMP_H

#include "Image.h"

#define BMP_HEADER_SIZE         54
#define BMP_COLOR_TABLE_SIZE    1024
#define CUSTOM_IMG_SIZE         1024*1024

class BMP : public Image
{
public:
    BMP();
    BMP(Image a, unsigned char head[54], unsigned char ct[1024], int bd, char* name);
    static BMP addImages(BMP a, BMP b);
    static void compareFile(BMP a, BMP b);
    static void compareHeader(BMP a, BMP b, bool colortable = 1);

    void printHeader();
    void readFile(char* filename, colorType ctype = GRAYSCALE);
    void writeFile(char* filename);
    virtual ~BMP();

    unsigned char header[54];
    unsigned char colorTable[1024];
    int bitDepth;

protected:

private:

};

#endif // BMP_H
