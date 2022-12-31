#include <stdio.h>
#include <stdlib.h>
#include "BMP.h"
#include <string>
#include <iostream>

using namespace std;


int main()
{
    BMP image = BMP();
    image.readFile("images/idk.bmp", RGB);
    image.printHeader();
    cout << image.size << endl;
    image.addPadding(1);
    image.writeFile("images/idk_blur.bmp");



    return 0;
}


//void imageReader()
