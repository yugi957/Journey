//#include "kernels.cuh"
//#include "general.h"
//#include "CUDAFunctions.cuh";
#include "BMP.h"
//#include "Image.h"
#include "Image.cuh"
using namespace std;

int main() {

    BMP image = BMP();
    image.readFile("images/idk.bmp", RGB);
    cout << image.size << endl;
    clock_t cpu_start, cpu_end;
    cpu_start = clock();
    image.Blur(11);
    cpu_end = clock();
    image.writeFile("images/idk_seq.bmp");
    printExecution("CPU execution", cpu_start, cpu_end);
    image.readFile("images/idk.bmp", RGB);
    image.parallelBlur(11);
    image.writeFile("images/idk_para.bmp");
    //image.writeData("images/idk_data.txt");

	return 0;
}

