//#include "kernels.cuh"
//#include "general.h"
//#include "CUDAFunctions.cuh";
//#include "BMP.h"
//#include "Image.h"
//#include "Image.cuh"
#include "MLP.h"
//#include "matConv.h"
#include <fstream>
#include "MNIST.h"
//#include "MLP.cuh"
using namespace std;

int main() {

	//train XOR 
	//cout << "\n\n-----------------TRAINED XOR EXAMPLE-----------------\n\n";
	//MultiLayerPerceptron MLP = MultiLayerPerceptron({2}, MSE);
	//MLP.addLayer(2, RELU);
	//MLP.addLayer(1, RELU);
	//cout << "Training Neural Network as XOR gate\n";
	//double MSE;
	//int epochs = 3000;
	//for (int i = 0;i < epochs;i++) {
	//	MSE = 0.0;
	//	MSE += MLP.bp({0,0},{0});
	//	MSE += MLP.bp({0,1},{1});
	//	MSE += MLP.bp({1,0},{1});
	//	MSE += MLP.bp({1,1},{0});
	//	MSE /= 4.0;
	//	if (i % 100 == 0) printf("Epoch %d: MSE = %f\n", i, MSE);
	//}

	////test XOR
	//printf("\n\nTESTING XOR GATE:::\n\n");
	//MLP.print_weights();

	//cout << "0 0: " << MLP.run({0,0})[0] << endl;
	//cout << "0 1: " << MLP.run({0,1})[0] << endl;
	//cout << "1 0: " << MLP.run({1,0})[0] << endl;
	//cout << "1 1: " << MLP.run({1,1})[0] << endl;


	//vector<Mat> images;
	//for (int i = 1;i < 10;i++) {
	//	//string file = (std::string("train/") + std::to_string(i) + ".jpg");
	//	//char* filename = new char[file.length() + 1];
	//	//strcpy(filename, file.c_str());
	//	char* file = createFilename("train/", to_string(i), ".jpg");
	//	Mat image = Mat();
	//	image = imread(file, IMREAD_GRAYSCALE);
	//	avgPool(&image, 9);
	//	prewittEdges(&image);
	//	namedWindow("Display frame", WINDOW_AUTOSIZE);
	//	imshow("Display frame", image);
	//	waitKey(0);
	//	images.push_back(image);
	//}

	vector<vector<double>> train_imgs, train_lbls, test_imgs, test_lbls;
	getMNIST(&train_imgs, &train_lbls, &test_imgs, &test_lbls);

	int s = train_imgs[0].size();
	cout << "\n\n----------IMAGE CLASSIFIER----------\n\n";

	printf("Size: %d\n", train_imgs[0].size());
	//MultiLayerPerceptron* mlp = new MultiLayerPerceptron({s}, CROSS_ENTROPY, 1, .001);
	//mlp->addLayer(512, SIGMOID);
	//mlp->addLayer(512, SIGMOID);
	//mlp->addLayer(10, SOFTMAX);
	MultiLayerPerceptron* mlp = new MultiLayerPerceptron({ s }, CROSS_ENTROPY, 1, .001);
	mlp->addLayer(512, SIGMOID);
	mlp->addLayer(512, SIGMOID);
	mlp->addLayer(10, SOFTMAX);
	cout << "Training Neural Network as Image Classifier...\n";
	double loss = 0.0;

	//for (int i = 0;i < 100;i++) {
	//	MSE = 0.0;
	//	for (int j = 0;j < 9;j++) {
	//		//cout << (double)j << endl;
	//		MSE += mlp->bp(train_imgs[j], train_lbls[j]);
	//	}
	//	MSE /= 9;
	//	cout << i << " : " << MSE << endl;
	//	if (i % 100 == 0)
	//		cout << "Epoch " << i << " MSE: " << MSE << endl;
	//}

	printf("Training on %d images and %d labels...\n", train_imgs.size(), train_lbls.size());

	vector<double> temp = { 0,0,0,0,0,0,0,0,0,0 };
	clock_t gpu_start, gpu_end;
	gpu_start = clock();
	for (int j = 0;j < 4;j++) {
		for (int i = 0;i < train_lbls.size();i++) {
			temp[train_lbls[i][0]] = 1;
			loss += mlp->bp(train_imgs[i], temp);
			temp[train_lbls[i][0]] = 0;
			//cout << i << " : " << MSE << endl;
			if (i % 250 == 0) {
				gpu_end = clock();
				cout << "Epoch " << i << " MSE: " << loss / 250 << endl;
				printExecution("Time taken", gpu_start, gpu_end);
				gpu_start = clock();
				loss = 0.0;
			}
		}
	}

	double correct = 0.0;
	for (int i = 0;i < test_lbls.size();i++) {
		vector<double> out = mlp->run(test_imgs[i]);
		int ans = 0;
		double top = 0.0;
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
	double accuracy = correct / (double)test_lbls.size();
	printf("\n\nAccuracy ====== %f\n.... %f correct out of %d tests\n", accuracy, correct, test_lbls.size());



	return 0;
}

