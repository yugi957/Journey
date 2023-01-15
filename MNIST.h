#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <vector>
using namespace std;

int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void getMNIST(vector<vector<double>>* train_imgs, vector<vector<double>>* train_lbls, vector<vector<double>>* test_imgs, vector<vector<double>>* test_lbls) {
    ifstream train_images("train-images.idx3-ubyte", ios::binary);
    ifstream train_labels("train-labels.idx1-ubyte", ios::binary);

    if (train_images.is_open() && train_labels.is_open()) {
        int magic_number = 0;
        int lbls_number = 0;
        int number_of_images = 0;
        int number_of_labels = 0;
        int n_rows = 0;
        int n_cols = 0;
        train_images.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        train_labels.read((char*)&lbls_number, sizeof(lbls_number));
        lbls_number = reverseInt(magic_number);
        printf("images magic number: %d\n", magic_number);
        printf("labels magic number: %d\n", lbls_number);
        train_images.read((char*)&number_of_images, sizeof(number_of_images));
        number_of_images = reverseInt(number_of_images);
        train_labels.read((char*)&number_of_labels, sizeof(number_of_labels));
        number_of_labels = reverseInt(number_of_labels);
        printf("number of images: %d\n", number_of_images);
        printf("number of images: %d\n", number_of_labels);
        train_images.read((char*)&n_rows, sizeof(n_rows));
        n_rows = reverseInt(n_rows);
        printf("rows: %d\n", n_rows);
        train_images.read((char*)&n_cols, sizeof(n_cols));
        n_cols = reverseInt(n_cols);
        printf("cols: %d\n", n_cols);
        printf("num of pixels: %d\n", n_cols * n_rows);
        //for (int i = 0;i < number_of_images;++i){
        unsigned char* data = (unsigned char*)malloc(sizeof(unsigned char) * n_cols * n_rows);
        for (int i = 0;i < number_of_images;i++) {
            train_imgs->push_back(vector<double>());
            train_lbls->push_back(vector<double>());
            for (int r = 0;r < n_rows;r++) {
                for (int c = 0;c < n_cols;c++) {
                    unsigned char temp = 0;
                    train_images.read((char*)&temp, sizeof(unsigned char));
                    //if(i > 60000) printf("Byte %d: %u\n", r * n_cols + c, temp);
                    data[r * n_cols + c] = temp;
                    (*train_imgs)[i].push_back((double)temp/255.0);
                }
            }
            unsigned char label = 0;
            train_labels.read((char*)&label, sizeof(unsigned char));
            (*train_lbls)[i].push_back((double)label);
            //file.read((char*)data, n_rows * n_cols);
            //Mat image = Mat(n_rows, n_cols, CV_8UC1, data);
            //imshow("Display frame", image);
            //waitKey(0);
            //imwrite("train/test.jpg", image);
        }
    }

    ifstream test_images("t10k-images.idx3-ubyte", ios::binary);
    ifstream test_labels("t10k-labels.idx1-ubyte", ios::binary);

    if (test_images.is_open() && test_labels.is_open()) {
        int magic_number = 0;
        int lbls_number = 0;
        int number_of_images = 0;
        int number_of_labels = 0;
        int n_rows = 0;
        int n_cols = 0;
        test_images.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        test_labels.read((char*)&lbls_number, sizeof(lbls_number));
        lbls_number = reverseInt(magic_number);
        printf("images magic number: %d\n", magic_number);
        printf("labels magic number: %d\n", lbls_number);
        test_images.read((char*)&number_of_images, sizeof(number_of_images));
        number_of_images = reverseInt(number_of_images);
        test_labels.read((char*)&number_of_labels, sizeof(number_of_labels));
        number_of_labels = reverseInt(number_of_labels);
        printf("number of images: %d\n", number_of_images);
        printf("number of images: %d\n", number_of_labels);
        test_images.read((char*)&n_rows, sizeof(n_rows));
        n_rows = reverseInt(n_rows);
        printf("rows: %d\n", n_rows);
        test_images.read((char*)&n_cols, sizeof(n_cols));
        n_cols = reverseInt(n_cols);
        printf("cols: %d\n", n_cols);
        printf("num of pixels: %d\n", n_cols * n_rows);
        //for (int i = 0;i < number_of_images;++i){
        unsigned char* data = (unsigned char*)malloc(sizeof(unsigned char) * n_cols * n_rows);
        for (int i = 0;i < number_of_images;i++) {
            test_imgs->push_back(vector<double>());
            test_lbls->push_back(vector<double>());
            for (int r = 0;r < n_rows;r++) {
                for (int c = 0;c < n_cols;c++) {
                    unsigned char temp = 0;
                    test_images.read((char*)&temp, sizeof(unsigned char));
                    //if(i > 60000) printf("Byte %d: %u\n", r * n_cols + c, temp);
                    data[r * n_cols + c] = temp;
                    (*test_imgs)[i].push_back((double)temp/255.0);
                }
            }
            unsigned char label = 0;
            test_labels.read((char*)&label, sizeof(unsigned char));
            (*test_lbls)[i].push_back((double)label);
            //file.read((char*)data, n_rows * n_cols);
            //Mat image = Mat(n_rows, n_cols, CV_8UC1, data);
            //imshow("Display frame", image);
            //waitKey(0);
            //imwrite("train/test.jpg", image);
        }
    }
}