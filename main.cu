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
#include "MLP.cuh"
#include "general.cuh"
using namespace std;

int main() {

	
    vector<vector<double>> train_imgs, train_lbls, test_imgs, test_lbls;

    getMNIST(&train_imgs, &train_lbls, &test_imgs, &test_lbls);
	
    int s = train_imgs[0].size();
	cout << "\n\n----------IMAGE CLASSIFIER----------\n\n";

	printf("Size: %d\n", train_imgs[0].size());
	//MultiLayerPerceptron* mlp = new MultiLayerPerceptron({s}, CROSS_ENTROPY, 1, .001);
	//mlp->addLayer(512, SIGMOID);
	//mlp->addLayer(512, SIGMOID);
	//mlp->addLayer(10, SOFTMAX);
	MultiLayerPerceptron* mlp = new MultiLayerPerceptron({ s }, CROSS_ENTROPY, 1, .01);
	mlp->addLayer(512, SIGMOID);
	mlp->addLayer(512, SIGMOID);
	mlp->addLayer(10, SOFTMAX);
	MultiLayerParatron* mlpara = new MultiLayerParatron({ s }, CROSS_ENTROPY, 1, .01);
	mlpara->addLayer(512, SIGMOID);
	mlpara->addLayer(512, SIGMOID);
	mlpara->addLayer(10, SOFTMAX);
	for (int i = 0;i < mlp->h_weights.size();i++) {
		for (int j = 0;j < mlp->h_weights[i].size();j++) {
			for (int k = 0;k < mlp->h_weights[i][j].size();k++) {
				mlpara->h_weights[i][j][k] = mlp->h_weights[i][j][k];
			}
		}
	}

	mlpara->finalize();
	cout << "Training Neural Network as Image Classifier...\n";
	double loss = 0.0;

    printf("Training on %d images and %d labels...\n", train_imgs.size(), train_lbls.size());

	vector<double> temp = { 0,0,0,0,0,0,0,0,0,0 };
	//vector<vector<double>> test = {{ 255/255.0,140/255.0,233/255.0,0/255.0,0/255.0,0/255.0,10/255.0,20/255.0,45/255.0,60/255.0}};
	//vector<vector<double>> test = {{ 255/255.0,140/255.0,233/255.0,0/255.0}};
	vector<vector<double>> train_encoders;
	for (int i = 0;i < train_lbls.size();i++) {
		temp[train_lbls[i][0]] = 1;
		train_encoders.push_back(temp);
		temp[train_lbls[i][0]] = 0;
	}
	double** d_train_imgs, **d_train_encoders, **d_test;
	//cudaAllocate2dOffVectorHostRef(&d_test, test);
	cudaAllocate2dOffVectorHostRef(&d_train_imgs, train_imgs);
	cudaAllocate2dOffVectorHostRef(&d_train_encoders, train_encoders);
	int size = test_imgs[0].size();
	
	clock_t gpu_start, gpu_end;
	vector<double> out;
	vector<double> o;

	out = mlp->Wrun(train_imgs[0]);
	//printf("\n\n\n");
	o = mlpara->getRun(d_train_imgs[0]);


	clock_t start, end;
	//start = clock();
	//o = mlpara->getRun(d_train_imgs[0]);
	//end = clock();
	//printExecution("GPU", start, end);
	//start = clock();
	//out = mlp->Wrun(train_imgs[0]);
	//end = clock();
	//printExecution("CPU", start, end);

	double l;
	//clock_t start, end;
	//start = clock();
	//for (int i = 0;i < 250;i++) {
	//	l = mlpara->bp(d_train_imgs[i], d_train_encoders[i]);
	//	//o = mlpara->getRun(d_train_imgs[i]);
	//	//loss = mlp->Wbp(train_imgs[i], train_encoders[i]);
	//	//out = mlp->Wrun(train_imgs[i]);
	//}
	//end = clock();
	//printExecution("GPU", start, end);
	//start = clock();
	//for (int i = 0;i < 250;i++) {
	//	loss = mlp->Wbp(train_imgs[i], train_encoders[i]);
	//}
	//end = clock();
	//printExecution("CPU", start, end);


	l = 0.0;
	loss = 0.0;

	


	gpu_start = clock();	
	for (int j = 0;j < 4;j++) {
		for (int i = 0;i < train_lbls.size();i++) {
			l += mlpara->bp(d_train_imgs[i], d_train_encoders[i]);
			//loss += mlp->Wbp(train_imgs[i], train_encoders[i]);
			//cout << i << " : " << MSE << endl;
			if (i % 250 == 0) {
				gpu_end = clock();
				//cout << "Epoch " << i << " MSE: " << loss / 250 << endl;
				cout << "Epoch " << i << " MSE: " << l / 250 << endl;
				l = 0.0;
				printExecution("Time taken", gpu_start, gpu_end);
				gpu_start = clock();
				loss = 0.0;
			}
		}
	}

	double** d_test_imgs = new double* [test_imgs.size()];
	cudaAllocate2dOffVectorHostRef(&d_test_imgs, test_imgs);

	double correct = 0.0;
    for (int i = 0;i < test_lbls.size();i++) {
		//vector<double> out = mlpara->getCleanRun(d_test_imgs[i]);
		vector<double> out = mlpara->getRun(d_test_imgs[i]);
		//vector<double> out = mlp->Wrun(test_imgs[i]);
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

