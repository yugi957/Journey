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
#include "general.h"
using namespace std;

int main() {

	
    vector<vector<double>> train_imgs, train_lbls, test_imgs, test_lbls;

    getMNIST(&train_imgs, &train_lbls, &test_imgs, &test_lbls);
	
    int s = train_imgs[0].size();
	int batchSize = 4;
	cout << "\n\n----------IMAGE CLASSIFIER----------\n\n";

	printf("Size: %d\n", train_imgs[0].size());
	//MultiLayerPerceptron* mlp = new MultiLayerPerceptron({s}, CROSS_ENTROPY, 1, .001);
	//mlp->addLayer(512, SIGMOID);
	//mlp->addLayer(512, SIGMOID);
	//mlp->addLayer(10, SOFTMAX);
	MultiLayerPerceptron* mlp = new MultiLayerPerceptron({ s }, CROSS_ENTROPY, 1, .01, batchSize);
	mlp->addLayer(512, SIGMOID);
	mlp->addLayer(512, SIGMOID);
	mlp->addLayer(10, SOFTMAX);
	MultiLayerParatron* mlpara = new MultiLayerParatron({ s }, CROSS_ENTROPY, 1, .01, batchSize);
	mlpara->addLayer(512, SIGMOID);
	mlpara->addLayer(512, SIGMOID);
	mlpara->addLayer(10, SOFTMAX);
	MultiLayerParatron* mlpar = new MultiLayerParatron({ s }, CROSS_ENTROPY, 1, .02, batchSize);
	mlpar->addLayer(512, SIGMOID);
	mlpar->addLayer(512, SIGMOID);
	mlpar->addLayer(10, SOFTMAX);
	for (int i = 0;i < mlp->h_weights.size();i++) {
		for (int j = 0;j < mlp->h_weights[i].size();j++) {
			for (int k = 0;k < mlp->h_weights[i][j].size();k++) {
				mlpara->h_weights[i][j][k] = mlp->h_weights[i][j][k];
				mlpar->h_weights[i][j][k] = mlp->h_weights[i][j][k];
			}
		}
	}

	mlpara->finalize();
	mlpar->finalize();
	mlp->finalize();
	cout << "Training Neural Network as Image Classifier...\n";
	double loss = 0.0;
	int numBatches = train_imgs.size() / batchSize;


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

	vector<vector<double>> x_batches = batchify(&train_imgs, batchSize);
	vector<vector<double>> y_batches = batchify(&train_encoders, batchSize);
	double** d_x_batches, **d_y_batches;
	cudaAllocate2dOffVectorHostRef(&d_x_batches, x_batches);
	cudaAllocate2dOffVectorHostRef(&d_y_batches, y_batches);
	//When shuffling batches, remember to free the batch list and then reallocate


	int size = test_imgs[0].size();
	
	clock_t gpu_start, gpu_end;
	//vector<double> out;
	//vector<double> o;


	clock_t start, end;

	double l;

	l = 0.0;
	//loss = 0.0;
	cudaMemcpy(&(*mlpara->d_weights_href[2]), &mlpara->h_weights[2][0][0], sizeof(double), cudaMemcpyHostToDevice);
	//vector<vector<double>> out, cleanOut, temparr;
	vector<double> out, cleanOut, temparr;


	int progressCheck = 250;

	//batch gradient descent
	gpu_start = clock();	
	for (int j = 0;j < 5;j++) {
		for (int i = 0;i < x_batches.size();i++) {
			loss += mlpar->aveBatchP(d_x_batches[i], d_y_batches[i]);

			if (i % (progressCheck / batchSize) == 0) {
				gpu_end = clock();
				cout << "Ground Example " << i << " error: " << l / (batchSize * (progressCheck /batchSize)) << endl;
				cout << "Epoch: " << j << ", Example " << i * batchSize << " error: " << loss / (progressCheck / batchSize) << endl;
				cout << endl;
				l = 0.0;
				printExecution("Time taken", gpu_start, gpu_end);
				gpu_start = clock();
				loss = 0.0;
			}
		}
		printf("Epoch %d completed\n", j);
		shuffleData(train_imgs, train_encoders);
		x_batches = batchify(&train_imgs, batchSize);
		y_batches = batchify(&train_encoders, batchSize);
		//THIS IS CAUSING MEMORY LEAKS
		cudaFree(&d_x_batches);
		cudaFree(&d_y_batches);
		cudaAllocate2dOffVectorHostRef(&d_x_batches, x_batches);
		cudaAllocate2dOffVectorHostRef(&d_y_batches, y_batches);
		gpu_end = clock();
		printExecution("Shuffle Time", gpu_start, gpu_end);
		gpu_start = clock();
	}

	//stochastic gradient descent
	//loss = 0.0;
	//gpu_start = clock();
	//for (int j = 0;j < 4;j++) {
	//	for (int i = 0;i < train_imgs.size();i++) {
	//		loss += mlpara->cleanerbp(d_train_imgs[i], d_train_encoders[i]);
	//		if (i % progressCheck == 0) {
	//			gpu_end = clock();
	//			cout << "Epoch: " << j << ", Example " << i << " error: " << loss / (progressCheck) << endl;
	//			cout << endl;
	//			l = 0.0;
	//			printExecution("Time taken", gpu_start, gpu_end);
	//			gpu_start = clock();
	//			loss = 0.0;
	//		}
	//	}
	//}

	double** d_test_imgs = new double* [test_imgs.size()];
	cudaAllocate2dOffVectorHostRef(&d_test_imgs, test_imgs);

	double correct = 0.0;
    for (int i = 0;i < test_lbls.size();i++) {
		//vector<double> out = mlpara->getCleanRun(d_test_imgs[i]);
		vector<double> out = mlpar->getCleanRun(d_test_imgs[i]);
		//mlpar->batchRun(d_x_batches[i]);
		//vector<double> out = mlp->Wrun(test_imgs[i]);
		int ans = 0;
		double top = 0.0;
		for (int i = 0;i < 10;i++) 
			if (out[i] > top) {
			top = out[i];
			ans = i;
		}
		//cout << "image " << i << ": [";
		//for (int i = 0;i < 10;i++) cout << out[i] << ", ";
		//cout << "] " << ans << " : " << test_lbls[i][0] << endl;
		if (ans == test_lbls[i][0]) correct++;
    }
	double accuracy = correct / (double)test_lbls.size();
	printf("\n\nAccuracy ====== %f\n.... %f correct out of %d tests\n", accuracy, correct, test_lbls.size());



	return 0;
}

