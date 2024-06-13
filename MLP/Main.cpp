//#include "kernels.cuh"
//#include "general.h"
//#include "CUDAFunctions.cuh";
//#include "BMP.h"
//#include "Image.h"
//#include "Image.cuh"
#include "Headers/MLP.h"
//#include "matConv.h"
#include <fstream>
#include "Headers/MNIST.h"
//#include "general.h"
//#include "MLP.cuh"
using namespace std;

int main() {

	vector<vector<vector<vector<float>>>> weight_data;
	vector<vector<float>> out_data;
	//train XOR 
	cout << "\n\n-----------------TRAINED XOR EXAMPLE-----------------\n\n";
	MultiLayerPerceptron MLP = MultiLayerPerceptron({2}, MSE, .1, .1, 1, 0.9);
	MLP.addLayer(2, SIGMOID);
	MLP.addLayer(1, SIGMOID);
	MLP.finalize();
	MLP.initializeWeights();
	cout << "Training Neural Network as XOR gate\n";
	float MSE;
	int epochs = 3000;
	//for (int i = 0;i < epochs;i++) {
	//	MSE = 0.0;
	//	MSE += MLP.Mbp({0,0},{0});
	//	MSE += MLP.Mbp({0,1},{1});
	//	MSE += MLP.Mbp({1,0},{1});
	//	MSE += MLP.Mbp({1,1},{0});
	//	MSE /= 4.0;
	//	if (i % 100 == 0) printf("Epoch %d: MSE = %f\n", i, MSE);
	//}

	////test XOR
	printf("\n\nTESTING XOR GATE:::\n\n");
	//MLP.print_weights();

	cout << "0 0: " << MLP.Wrun({0,0})[0] << endl;
	cout << "0 1: " << MLP.Wrun({0,1})[0] << endl;
	cout << "1 0: " << MLP.Wrun({1,0})[0] << endl;
	cout << "1 1: " << MLP.Wrun({1,1})[0] << endl;

	write4D(weight_data);
	write2D(out_data);

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

	vector<vector<float>> train_imgs, train_lbls, test_imgs, test_lbls;
	getMNIST(&train_imgs, &train_lbls, &test_imgs, &test_lbls);
	int s = train_imgs[0].size();
	int batchSize = 20;
	//vector<vector<vector<float>>> batches = batchify(&train_imgs, batchSize);
	//vector<vector<vector<float>>> batch_labels = batchify(&train_lbls, batchSize);

	cout << "\n\n----------IMAGE CLASSIFIER----------\n\n";

	printf("Size: %d\n", train_imgs[0].size());
	MultiLayerPerceptron* mlp = new MultiLayerPerceptron({ s }, CROSS_ENTROPY, .1, 1, 1, .9);
	mlp->addLayer(500, RELU);
	mlp->addLayer(300, RELU);
	mlp->addLayer(10, SOFTMAX);
	MultiLayerPerceptron* testlp = new MultiLayerPerceptron({ s }, CROSS_ENTROPY, 1, .01);
	testlp->addLayer(512, SIGMOID);
	testlp->addLayer(512, SIGMOID);
	testlp->addLayer(10, SOFTMAX);
	//cout << "Training Neural Network as Image Classifier...\n";
	testlp->h_weights = mlp->h_weights;
	compare3D(testlp->h_weights, mlp->h_weights);
	mlp->finalize();

	printf("Training on %d images and %d labels...\n", train_imgs.size(), train_lbls.size());
	//printf("Batch Size: %d, num_batches: %d\n", batchSize, batches.size());

	int progressCheck = 50;

	vector<vector<float>> labels = autoencode(train_lbls, 10);

	mlp->train(train_imgs, labels, 4, 50);

	float correct = 0.0;
	for (int i = 0;i < test_lbls.size();i++) {
		vector<float> out = mlp->Wrun(test_imgs[i]);
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

