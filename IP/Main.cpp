#include "../general.h"
#include "../general.h"
#include <opencv2/opencv.hpp>
//#include "Headers/ImageProcessing.h"
#include "../MLP/Headers/MLP.h"
#include "../MLP/Headers/MNIST.h"
#include <fstream>
using namespace cv;

int main() {

	MultiLayerPerceptron cnn = MultiLayerPerceptron({}, MSE, 1, .01, 0, 0);
	cnn.addLayer(4 * 4 * 1, RELU, vec3(4, 4, 1));
	cnn.addConv(2, 0, 1, 1, RELU);
	cnn.addMaxPool(2, 0, 1);
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
	vector<float> y_example = { 8, 9, 9, 7 };
	cnn.forward_conv(example);
	cnn.backward_conv(example, y_example);


	//MultiLayerPerceptron LeNet = MultiLayerPerceptron({}, CROSS_ENTROPY, 1, .1, 1, 0);
	//LeNet.addLayer(28*28*1, RELU, vec3(28, 28, 1));
	//LeNet.addConv(5, 2, 1, 6, RELU);
	//LeNet.addMaxPool(2, 0, 2);
	//LeNet.addConv(5, 0, 1, 16, RELU);
	//LeNet.addMaxPool(2, 0, 2);
	//LeNet.addLayer(120, RELU);
	//LeNet.addLayer(84, RELU);
	//LeNet.addLayer(10, SOFTMAX);
	//LeNet.finalize();
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

	MultiLayerPerceptron* mlp = new MultiLayerPerceptron({}, CROSS_ENTROPY, 1, .01, 0, 0);
	mlp->addLayer(784, RELU);
	mlp->addLayer(500, RELU);
	mlp->addLayer(300, RELU);
	mlp->addLayer(10, SOFTMAX);
	mlp->finalize();
	MultiLayerPerceptron* test = new MultiLayerPerceptron({}, CROSS_ENTROPY, 1, .01, 0, 0);
	test->addLayer(784, RELU);
	test->addLayer(500, RELU);
	test->addLayer(300, RELU);
	test->addLayer(10, SOFTMAX);
	test->finalize();

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
			l += mlp->Wbp(train_imgs[i], train_encoders[i]);
			loss += test->backward_conv(train_imgs[i], train_encoders[i]);
			//loss += LeNet.backward_conv(train_imgs[i], train_encoders[si]);
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
	for (int i = 0;i < test_lbls.size();i++) {
		vector<float> out = test->forward_conv(test_imgs[i]);
		int ans = 0;
		float top = 0.0;
		for (int i = 0;i < 10;i++)
			if (out[i] > top) {
				top = out[i];
				ans = i;
			}
		cout << "image " << i << ": [";
		for (int i = 0;i < 10;i++) cout << out[i] << ", ";
		cout << "] " << ans << " : " << test_lbls[i][0] << endl;
		if (ans == test_lbls[i][0]) correct++;
	}
	float accuracy = correct / (float)test_lbls.size();
	printf("\n\nAccuracy ====== %f\n.... %f correct out of %d tests\n", accuracy, correct, test_lbls.size());
	return 0;
}