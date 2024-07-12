#include "../general.h"
#include <opencv2/opencv.hpp>
//#include "Headers/ImageProcessing.h"
#include "../CUDA/Headers/general.cuh"
#include "../MLP/Headers/MLP.h"
#include "../MLP/Headers/MLP.cuh"
#include "../MLP/Headers/MNIST.h"
#include "../MLP/data/cifar10_reader.hpp"
#include <fstream>
using namespace cv;

int main() {

	vector<vector<float>> cifar_train_images;
	vector<vector<float>> cifar_train_encoders;
	vector<vector<float>> cifar_test_images;
	vector<vector<float>> cifar_test_encoders;

	if (true) {
		auto dataset = cifar::read_dataset<std::vector, std::vector, float, float>();
		for (auto& image : dataset.training_images) {
			std::transform(image.begin(), image.end(), image.begin(),
				[](float pixel) { return pixel / 255.0f; });
		}
		for (auto& image : dataset.test_images) {
			std::transform(image.begin(), image.end(), image.begin(),
				[](float pixel) { return pixel / 255.0f; });
		}

		cifar_train_images = dataset.training_images;
		cifar_train_encoders = autoencode(dataset.training_labels, 10);
		cifar_test_images = dataset.test_images;
		cifar_test_encoders = autoencode(dataset.training_labels, 10);
	}
	
	MultiLayerPerceptron cifarNet = MultiLayerPerceptron({}, CROSS_ENTROPY, 1, .01, 0, 0);
	cifarNet.addLayer(32*32*3, RELU, vec3(32,32,3));
	cifarNet.addConv(3, 0, 1, 32, RELU);
	cifarNet.addMaxPool(2, 0, 2);
	cifarNet.addConv(5, 0, 1, 64, RELU);
	cifarNet.addMaxPool(3, 0, 3);
	cifarNet.addConv(3, 0, 1, 64, RELU);
	cifarNet.addLayer(64, RELU);
	cifarNet.addLayer(10, SOFTMAX);
	cifarNet.finalize();
	MultiLayerParatron gpuCifarNet = MultiLayerParatron({}, CROSS_ENTROPY, 1, .01, 0, 0);
	gpuCifarNet.addLayer(32 * 32 * 3, RELU, vec3(32, 32, 3));
	gpuCifarNet.addConv(3, 1, 1, 32, RELU);
	gpuCifarNet.addConv(3, 1, 1, 32, RELU);
	gpuCifarNet.addMaxPool(2, 0, 2);
	gpuCifarNet.addLayer(128, RELU);
	gpuCifarNet.addLayer(10, SOFTMAX);
	gpuCifarNet.finalize();
	MultiLayerPerceptron LeNet = MultiLayerPerceptron({}, CROSS_ENTROPY, 1, .01, 0, 0);
	LeNet.addLayer(28 * 28 * 1, RELU, vec3(28, 28, 1));
	LeNet.addConv(5, 2, 1, 6, RELU);
	LeNet.addMaxPool(2, 0, 2);
	LeNet.addConv(5, 0, 1, 16, RELU);
	LeNet.addMaxPool(2, 0, 2);
	LeNet.addLayer(120, RELU);
	LeNet.addLayer(80, RELU);
	LeNet.addLayer(10, SOFTMAX);
	LeNet.finalize();
	MultiLayerParatron gpuLeNet = MultiLayerParatron({}, CROSS_ENTROPY, 1, .01, 0, 0);
	gpuLeNet.addLayer(28 * 28 * 1, RELU, vec3(28, 28, 1));
	gpuLeNet.addConv(5, 2, 1, 6, RELU);
	gpuLeNet.addMaxPool(2, 0, 2);
	gpuLeNet.addConv(5, 0, 1, 16, RELU);
	gpuLeNet.addMaxPool(2, 0, 2);
	gpuLeNet.addLayer(120, RELU);
	gpuLeNet.addLayer(80, SIGMOID);
	gpuLeNet.addLayer(10, SOFTMAX);
	gpuLeNet.finalize();
	MultiLayerPerceptron CNN = MultiLayerPerceptron({}, CROSS_ENTROPY, 1, .01, 0, 0);
	CNN.addLayer(28*28*1, RELU, vec3(28,28,1));
	CNN.addConv(3, 0, 1, 4, RELU);
	CNN.addMaxPool(2, 0, 2);
	CNN.addConv(3, 0, 1, 4, RELU);
	CNN.addMaxPool(2, 0, 2);
	CNN.addLayer(100, RELU);
	CNN.addLayer(10, SOFTMAX);
	CNN.finalize();
	MultiLayerParatron test = MultiLayerParatron({}, CROSS_ENTROPY, 1, .01, 0, 0);
	test.addLayer(28 * 28 * 1, RELU, vec3(28, 28, 1));
	test.addConv(3, 0, 1, 4, RELU);
	test.addMaxPool(2, 0, 2);
	test.addConv(3, 0, 1, 4, RELU);
	test.addMaxPool(2, 0, 2);
	test.addLayer(100, RELU);
	test.addLayer(10, SOFTMAX);
	test.finalize();
	test.conv_weights = CNN.conv_weights;
	vector<vector<float>> convs2d;
	for (int i = 0;i < CNN.conv_weights.size();i++) {
		convs2d.push_back(flatten4D(CNN.conv_weights[i]));
	}
	cudaAllocate2dOffVectorHostRef(&test.d_conv_weights_href, convs2d);
	vector<vector<float>> train_imgs, train_lbls, test_imgs, test_lbls;
	getMNIST(&train_imgs, &train_lbls, &test_imgs, &test_lbls);
	vector<float> temp = { 0,0,0,0,0,0,0,0,0,0 };
	vector<vector<float>> train_encoders;
	for (int i = 0;i < train_lbls.size();i++) {
		temp[train_lbls[i][0]] = 1;
		train_encoders.push_back(temp);
		temp[train_lbls[i][0]] = 0;
	}
	//cout << "Training Neural Network as Image Classifier...\n";
	int batchSize = 1;
	float** d_train_imgs, ** d_train_encoders, ** d_test, ** d_cifar_imgs, ** d_cifar_encoders;
	//cudaAllocate2dOffVectorHostRef(&d_train_imgs, train_imgs);
	//cudaAllocate2dOffVectorHostRef(&d_train_encoders, train_encoders);
	cudaAllocate2dOffVectorHostRef(&d_cifar_imgs, cifar_train_images);
	cudaAllocate2dOffVectorHostRef(&d_cifar_encoders, cifar_train_encoders);
	//vector<vector<float>> x_batches = batchify(&train_imgs, batchSize);
	//vector<vector<float>> y_batches = batchify(&train_encoders, batchSize);
	//float** d_x_batches, ** d_y_batches;
	//cudaAllocate2dOffVectorHostRef(&d_x_batches, x_batches);
	//cudaAllocate2dOffVectorHostRef(&d_y_batches, y_batches);

	int epochs = 1;
	float loss = 0.0;
	float l = 0.0;
	int progressCheck = 50;
	clock_t gpu_start, gpu_end;
	gpu_start = clock();
	for (int j = 0;j < epochs;j++) {
		for (int i = 0;i < train_encoders.size();i++) {
			//loss += CNN.backward_conv(train_imgs[i], train_encoders[i]);
			//loss += LeNet.backward_conv(train_imgs[i], train_encoders[i]);
			//loss += cifarNet.backward_conv(cifar_train_images[i], cifar_train_encoders[i]);
			//l += test.backward_conv(d_train_imgs[i], d_train_encoders[i]);
			l += gpuCifarNet.backward_conv(d_cifar_imgs[i], d_cifar_encoders[i]);
			//l += gpuLeNet.backward_conv(d_train_imgs[i], d_train_encoders[i]);
			//compare5D(test.conv_weights, CNN.conv_weights);
			//compare5D(test.conv_gradient, CNN.conv_gradient);
			if (i % progressCheck == 0) {
				gpu_end = clock();
				cout << "ground truth example: " << i << " error: " << loss / progressCheck << endl;
				cout << "test example: " << i << " error: " << l / progressCheck << endl;
				printExecution("Time taken", gpu_start, gpu_end);
				gpu_start = clock();
				loss = 0.0;
				l = 0.0;
			}
		}
	}

	float** d_test_imgs;
	float** d_cifar_test_imgs;
	//cudaAllocate2dOffVectorHostRef(&d_test_imgs, test_imgs);
	cudaAllocate2dOffVectorHostRef(&d_cifar_test_imgs, cifar_test_images);
	float correct = 0.0;
	float test_correct = 0.0;
	for (int i = 0;i < test_lbls.size();i++) {
		if (i % 1000 == 0) cout << i << endl;
		//vector<float> out = LeNet.forward_conv(test_imgs[i]);
		//vector<float> out(10,0);
		//vector<float> out = gpuCifarNet.getForwardConv(d_test_imgs[i]);
		vector<float> out = cifarNet.forward_conv(cifar_test_images[i]);
		//vector<float> test_out = gpuLeNet.getForwardConv(d_test_imgs[i]);
		vector<float> test_out(10,0);
		int ans = distance(out.begin(), max_element(out.begin(), out.end()));
		int test_ans = distance(test_out.begin(), max_element(test_out.begin(), test_out.end()));
		if (ans == test_lbls[i][0]) correct++;
		//if (ans == *max_element(cifar_test_encoders[i].begin(), cifar_test_encoders[i].end())) correct++;
		if (test_ans == test_lbls[i][0]) test_correct++;
	}
	float accuracy = correct / (float)test_lbls.size();
	float test_accuracy = test_correct / (float)test_lbls.size();
	printf("\n\nAccuracy ====== %f\n.... %f correct out of %d tests\n", accuracy, correct, test_lbls.size());
	printf("\n\nAccuracy ====== %f\n.... %f correct out of %d tests\n", test_accuracy, test_correct, test_lbls.size());
	return 0;
}