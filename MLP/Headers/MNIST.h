#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <vector>
using namespace std;

int reverseInt(int i);

void getMNIST(vector<vector<double>>* train_imgs, vector<vector<double>>* train_lbls, vector<vector<double>>* test_imgs, vector<vector<double>>* test_lbls);