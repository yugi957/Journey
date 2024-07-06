#include "../general.h"
#include "../general.h"
#include <opencv2/opencv.hpp>
//#include "Headers/ImageProcessing.h"
#include "../MLP/Headers/MLP.h"
#include "../MLP/Headers/MNIST.h"
#include "../MLP/Headers/MLP.cuh"
#include "../CUDA/Headers/general.cuh"
#include <fstream>
using namespace cv;

int main() {

	MultiLayerPerceptron cnn = MultiLayerPerceptron({}, MSE, 1, .01, 0, 0);
	cnn.addLayer(4 * 4 * 1, RELU, vec3(4, 4, 1));
	cnn.addConv(2, 0, 1, 1, RELU);
	cnn.addConv(2, 0, 1, 1, RELU);
	vector<float> example = { 1, 0, 2, 3,
							  4, 6, 6, 8,
							  3, 1, 1, 0,
							  1, 2, 2, 4
	};
	cnn.finalize();
	cnn.conv_weights[0][0][0][0][0] = 1;
	cnn.conv_weights[0][0][0][1][0] = 0;
	cnn.conv_weights[0][0][1][0][0] = 0;
	cnn.conv_weights[0][0][1][1][0] = 1;
	cnn.conv_weights[1][0][0][0][0] = -1;
	cnn.conv_weights[1][0][0][1][0] = 1;
	cnn.conv_weights[1][0][1][0][0] = 0;
	cnn.conv_weights[1][0][1][1][0] = 0;
	vector<float> y_example = { 8, 9, 2, 2 };
	cnn.forward_conv(example);
	cnn.backward_conv(example, y_example);
	cnn.backward_conv(example, y_example);


	MultiLayerPerceptron LeNet = MultiLayerPerceptron({}, CROSS_ENTROPY, 1, .01, 1, 0);
	LeNet.addLayer(28*28*1, RELU, vec3(28, 28, 1));
	LeNet.addConv(5, 2, 1, 6, RELU);
	LeNet.addMaxPool(2, 0, 2);
	LeNet.addConv(5, 0, 1, 16, RELU);
	LeNet.addMaxPool(2, 0, 2);
	LeNet.addLayer(120, RELU);
	LeNet.addLayer(84, RELU);
	LeNet.addLayer(10, SOFTMAX);
	LeNet.finalize();
	MultiLayerPerceptron CNN = MultiLayerPerceptron({}, CROSS_ENTROPY, 1, .01, 1, 0);
	CNN.addLayer(28*28*1, RELU, vec3(28,28,1));
	CNN.addConv(3, 1, 1, 6, RELU);
	CNN.addLayer(100, RELU);
	CNN.addLayer(10, SOFTMAX);
	CNN.finalize();
	vector<vector<float>> train_imgs, train_lbls, test_imgs, test_lbls;
	getMNIST(&train_imgs, &train_lbls, &test_imgs, &test_lbls);
	vector<float> temp = { 0,0,0,0,0,0,0,0,0,0 };
	vector<vector<float>> train_encoders;
	for (int i = 0;i < train_lbls.size();i++) {
		temp[train_lbls[i][0]] = 1;
		train_encoders.push_back(temp);
		temp[train_lbls[i][0]] = 0;
	}
	//LeNet.forward_conv(train_imgs[0]);
	//float loss = LeNet.backward_conv(train_imgs[0], train_encoders[0]);
	cout << sum << endl << endl;
	//cout << "Training Neural Network as Image Classifier...\n";

	int batchSize = 1;
	MultiLayerParatron* mlpar = new MultiLayerParatron({ 784 }, CROSS_ENTROPY, 1, .01, 0, batchSize);
	mlpar->addLayer(500, RELU);
	mlpar->addLayer(300, RELU);
	mlpar->addLayer(10, SOFTMAX);
	mlpar->finalize();
	float** d_train_imgs, ** d_train_encoders, ** d_test;
	cudaAllocate2dOffVectorHostRef(&d_train_imgs, train_imgs);
	cudaAllocate2dOffVectorHostRef(&d_train_encoders, train_encoders);
	//vector<vector<float>> x_batches = batchify(&train_imgs, batchSize);
	//vector<vector<float>> y_batches = batchify(&train_encoders, batchSize);
	//float** d_x_batches, ** d_y_batches;
	//cudaAllocate2dOffVectorHostRef(&d_x_batches, x_batches);
	//cudaAllocate2dOffVectorHostRef(&d_y_batches, y_batches);

	int epochs = 5;
	float loss = 0.0;
	float l = 0.0;
	int progressCheck = 50;
	clock_t gpu_start, gpu_end;
	gpu_start = clock();
	for (int j = 0;j < epochs;j++) {
		for (int i = 0;i < train_encoders.size();i++) {
			//if (i == 900) LeNet.eta = .01;
			//temp[train_lbls[i][0]] = 1;
			//compare3D(MLP.h_weights, moMLP.h_weights);
			//loss += LeNet.backward_conv(train_imgs[i], train_encoders[i])
			//l += mlp->Wbp(train_imgs[i], train_encoders[i]);
			//loss += LeNet.backward_conv(train_imgs[i], train_encoders[i]);
			loss += CNN.backward_conv(train_imgs[i], train_encoders[i]);
			//compare2D(MLP.outputs, test.outputs);
			//compareHtoConvWeight(MLP.h_weights, test.conv_weights);
			//temp[train_lbls[i][0]] = 0;
			//cout << i << " : " << MSE << endl;
			if (i % progressCheck == 0) {
				gpu_end = clock();
				cout << "ground truth example: " << i << " error: " << l / progressCheck << endl;
				cout << "test example: " << i << " error: " << loss / progressCheck << endl;
				printExecution("Time taken", gpu_start, gpu_end);
				gpu_start = clock();
				loss = 0.0;
				l = 0.0;
			}
		}
	}

	float correct = 0.0;
	float test_correct = 0.0;
	for (int i = 0;i < test_lbls.size();i++) {
		vector<float> out = LeNet.Wrun(test_imgs[i]);
		//vector<float> test_out = LeNet.forward_conv(test_imgs[i]);
		vector<float> test_out = CNN.forward_conv(test_imgs[i]);
		int ans = 0;
		int test_ans = 0;
		float top = 0.0;
		float test_top = 0.0;
		for (int i = 0;i < 10;i++) {
			if (out[i] > top) {
				top = out[i];
				ans = i;
			}
			if (test_out[i] > top) {
				test_top = out[i];
				test_ans = i;
			}
		}
		if (ans == test_lbls[i][0]) correct++;
		if (test_ans == test_lbls[i][0]) test_correct++;
	}
	float accuracy = correct / (float)test_lbls.size();
	float test_accuracy = test_correct / (float)test_lbls.size();
	printf("\n\nAccuracy ====== %f\n.... %f correct out of %d tests\n", accuracy, correct, test_lbls.size());
	printf("\n\nAccuracy ====== %f\n.... %f correct out of %d tests\n", test_accuracy, test_correct, test_lbls.size());
	return 0;
}