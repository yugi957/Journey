#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <vector>
using namespace std;

int reverseInt(int i);

void getMNIST(vector<vector<float>>* train_imgs, vector<vector<float>>* train_lbls, vector<vector<float>>* test_imgs, vector<vector<float>>* test_lbls);
void getFashionMNIST(vector<vector<float>>* train_imgs, vector<vector<float>>* train_lbls, vector<vector<float>>* test_imgs, vector<vector<float>>* test_lbls);