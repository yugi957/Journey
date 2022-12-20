#include <stdio.h>
#include <stdlib.h>
#include "BMP.h"
#include <string>


int main()
{
    BMP image = BMP();
    image.readFile("images/fruits.bmp", RGB);
    image.Blur(7.0);
    image.writeFile("images/fruits_sharp.bmp");



    return 0;
}


//void imageReader()
