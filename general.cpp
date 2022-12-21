#include "general.h"

void generateRandArray(int* arr, int size) {
	time_t t;
	srand((unsigned)time(&t));
	for (int i = 0;i < size;i++) {
		arr[i] = (int)(rand() & 0xff);
	}
}

void sum_array_cpu(int* a, int* b, int* c, int size) {

	for (int i = 0;i < size;i++) {
		c[i] = a[i] + b[i];
	}
}

void sum_arrays_cpu(int* a, int* b, int* c, int* d, int size) {

	for (int i = 0;i < size;i++) {
		d[i] = a[i] + b[i] + c[i];
	}
}

void compare_arrays(int* a, int* b, int size) {
	for (int i = 0;i < size;i++) {
		if (a[i] != b[i]) {
			printf("%d :: %d\n", a[i], b[i]);
			printf("ARRAYS ARE DIFFERENT\n");
			return;
		}
	}
	printf("SUCCESS Arrays are the same\n");
}