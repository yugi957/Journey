//#include "kernels.cuh"
#include "general.h"
//#include "CUDAFunctions.cuh";
#include "BMP.h"
#include "Image.h"
#include "Image.cuh"
#include "MLP.h"
#include "matConv.h"
using namespace std;

int main() {

	//train XOR 
	//cout << "\n\n-----------------TRAINED XOR EXAMPLE-----------------\n\n";
	//MultiLayerPerceptron mlp = MultiLayerPerceptron({ 2,2,1 });
	//cout << "Training Neural Network as XOR gate\n";
	//double MSE;
	//int epochs = 3000;
	//for (int i = 0;i < epochs;i++) {
	//	MSE = 0.0;
	//	MSE += mlp.bp({0,0},{0});
	//	MSE += mlp.bp({0,1},{1});
	//	MSE += mlp.bp({1,0},{1});
	//	MSE += mlp.bp({1,1},{0});
	//	MSE /= 4.0;
	//	if (i % 100 == 0) printf("Epoch %d: MSE = %f\n", i, MSE);
	//}

	////test XOR
	//printf("\n\nTESTING XOR GATE:::\n\n");
	//mlp.print_weights();

	//cout << "0 0: " << mlp.run({0,0})[0] << endl;
	//cout << "0 1: " << mlp.run({0,1})[0] << endl;
	//cout << "1 0: " << mlp.run({1,0})[0] << endl;
	//cout << "1 1: " << mlp.run({1,1})[0] << endl;


	vector<Mat> images;
	for (int i = 1;i < 10;i++) {
		//string file = (std::string("train/") + std::to_string(i) + ".jpg");
		//char* filename = new char[file.length() + 1];
		//strcpy(filename, file.c_str());
		char* file = createFilename("train/", to_string(i), ".jpg");
		Mat image = Mat();
		image = imread(file, IMREAD_GRAYSCALE);
		avgPool(&image, 9);
		prewittEdges(&image);
		namedWindow("Display frame", WINDOW_AUTOSIZE);
		imshow("Display frame", image);
		waitKey(0);
		images.push_back(image);
	}
	
	vector<vector<double>> inputs;
	for (int i = 0;i < images.size();i++) {
		inputs.push_back(vector<double>());
		int s = getSize(images[i].rows, images[i].cols, images[i].channels());
		for (int j = 0;j < s;j++) {
			inputs[i].push_back((double)images[i].data[j]);
		}
	}
	for (int i = 0;i < images.size();i++) {
		int s = (int)inputs[i].size();
	}
	int s = (int)inputs[0].size();
	
	
	
	cout << "\n\n----------IMAGE CLASSIFIER----------\n\n";

	printf("Size: %d\n", inputs[0].size());
	MultiLayerPerceptron* mlp = new MultiLayerPerceptron({ s,s / 3, s / 4, 9 }, 1, .25);
	cout << "Training Neural Network as Image Classifier...\n";
	double MSE;


	vector<vector<double>> labels;
	for (int i = 1;i < 10; i++) {
		labels.push_back(vector<double>());
		for (int j = 1;j < 10;j++) {
			if (i == j) labels[i - 1].push_back(1.0);
			else labels[i - 1].push_back(0.0);
			cout << labels[i - 1][j - 1] << " ";
		}
		cout << endl;
	}

	for (int i = 0;i < 100;i++) {
		MSE = 0.0;
		for (int j = 0;j < 9;j++) {
			//cout << (double)j << endl;
			MSE += mlp->bp(inputs[j], labels[j]);
		}
		MSE /= 9;
		cout << i << " : " << MSE << endl;
		if (i % 100 == 0)
			cout << "Epoch " << i << " MSE: " << MSE << endl;
	}





	return 0;
}

