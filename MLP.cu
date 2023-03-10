#include "MLP.cuh"
#include "CUDAFunctions.cuh"


//perceptron class methods:


//return new perceptron with number of inputs + 1 bias
Paratron::Paratron(int inputs, activation_function func, double bias) {
    this->bias = bias;
    this->A_F = func;
    this->inputSize = inputs;
    weights.resize(inputs + 1);
    generate(weights.begin(), weights.end(), frand);
    cudaMalloc((void**)&this->d_weights, sizeof(double) * (inputs + 1));
    cudaMalloc((void**)&this->d_vectorInput, sizeof(double) * (inputs));
    double* temp = &weights[0];
    cudaMemcpy(this->d_weights, temp, sizeof(double) * (inputs + 1), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&this->d_bufferProduct, sizeof(double) * inputs);

}

//Run the perceptron with x as input
double Paratron::run(vector<double> x) {
    cudaMemcpy(this->d_vectorInput, &x[0], sizeof(double) * inputSize, cudaMemcpyHostToDevice);
    double sum = innerProduct(d_vectorInput, d_weights, d_bufferProduct, inputSize);
    sum += bias * weights[inputSize];
    return activation(sum);

}

double Paratron::run(double* d_x) {
    //We will add bias weight * bias separately
    double sum;
    sum = innerProduct(d_x, d_weights, d_bufferProduct, inputSize - 1);
    sum += bias * weights[inputSize];
    return activation(sum);
}

//MLP class methods:
MultiLayerParatron::MultiLayerParatron(vector<int> CIL, loss_function func, double bias, double eta) {
    this->cells_in_layer = CIL;
    this->L_F = func;
    this->bias = bias;
    this->eta = eta;

    for (int i = 0;i < cells_in_layer.size();i++) {
        outputs.push_back(vector<double>(cells_in_layer[i], 0.0));
        //error_terms.push_back(vector<double>(cells_in_layer[i], 0.0));
        network.push_back(vector<Paratron>());
        //clock_t gpu_start, gpu_end;
        if (i > 0) { //input layer has no neurons
            for (int j = 0;j < cells_in_layer[i];j++) {
                //gpu_start = clock();
                network[i].push_back(Paratron(cells_in_layer[i - 1], SIGMOID, bias));
                //gpu_end = clock();
                //printf("%d : %d ", cells_in_layer[i], j);
                //printExecution("Loop", gpu_start, gpu_end);
            }
        }
    }
}

void MultiLayerParatron::finalize() {
    int* weightLayerOffsets = new int[cells_in_layer.size() - 1];
    weightLayerOffsets[0] = 0;
    for (int i = 1;i < cells_in_layer.size()-1;i++) {
        weightLayerOffsets[i] = weightLayerOffsets[i - 1] + (cells_in_layer[i] * (cells_in_layer[i-1] + 1));
    }
    cudaMalloc((void**)&d_weightLayerOffsets, sizeof(int) * (cells_in_layer.size() - 1));
    cudaMemcpy(d_weightLayerOffsets, weightLayerOffsets, sizeof(int) * (cells_in_layer.size() - 1), cudaMemcpyHostToDevice);

    int* outputLayerOffsets = new int[cells_in_layer.size()];
    outputLayerOffsets[0] = 0;
    for (int i = 1;i < cells_in_layer.size();i++) {
        outputLayerOffsets[i] = outputLayerOffsets[i - 1] + cells_in_layer[i - 1];
    }
    cudaMalloc((void**)&d_outputLayerOffsets, sizeof(int) * (cells_in_layer.size()));
    cudaMemcpy(d_outputLayerOffsets, outputLayerOffsets, sizeof(int) * (cells_in_layer.size()), cudaMemcpyHostToDevice);
    int size = sizeof(double);
    cudaMalloc((void**)&d_CIL, size * cells_in_layer.size());
    cudaMemcpy(d_CIL, &cells_in_layer[0], size * cells_in_layer.size(), cudaMemcpyHostToDevice);
    cudaAllocate2dOffVector(&d_outputs, outputs);
    cudaAllocate2dOffVectorHostRef(&d_outputs_href, outputs);
    cudaAllocate3dOffVector(&d_weights, h_weights);
    for (int i = 1;i < cells_in_layer.size();i++) {
        weight_lengths.push_back(vector<int>());
        for (int j = 0;j < cells_in_layer[i];j++) {
            weight_lengths[i-1].push_back(cells_in_layer[i - 1]+1);
        }
    }
    cudaAllocate3dOffVectorHostRef(&d_weights_href, h_weights);
    activation_function* h_A_Fs = new activation_function[network.size()];
    for (int i = 1;i < network.size();i++) {
        h_A_Fs[i] = network[i][0].A_F;
    }
    cudaMalloc((void**)&d_A_Fs, sizeof(activation_function) * network.size());
    cudaMemcpy(d_A_Fs, h_A_Fs, sizeof(activation_function) * network.size(), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_L_F, sizeof(loss_function));
    cudaMemcpy(d_L_F, &L_F, sizeof(loss_function), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_loss, sizeof(double));
    cudaMalloc((void**)&d_eta, sizeof(double));
    cudaMemcpy(d_eta, &eta, sizeof(double), cudaMemcpyHostToDevice);
    cudaAllocate2dOffVector(&d_error_terms, error_terms);
    cudaAllocate2dOffVectorHostRef(&d_error_terms_href, error_terms);
    //cudaCopy3dBackToVector(&d_weights, weight_lengths);

}

void MultiLayerParatron::addLayer(int CIL, activation_function func) {
    if (cells_in_layer.size() > 0) { //means input layer is done
        int LL = cells_in_layer.size() - 1; //Last Layer index before new layer
        cells_in_layer.push_back(CIL);
        outputs.push_back(vector<double>(CIL, 0.0));
        error_terms.push_back(vector<double>(CIL, 0.0));
        h_weights.push_back(vector<vector<double>>(CIL, vector<double>(cells_in_layer[LL]+1, 0.0)));
        for (int i = 0;i < CIL;i++) {
            generate(h_weights[LL][i].begin(), h_weights[LL][i].end(), frand);
        }
        network.push_back(vector<Paratron>());
        for (int i = 0;i < CIL;i++) {
            network[LL + 1].push_back(Paratron(cells_in_layer[LL], func, bias));
        }
    }
    else printf("INITIALIZE NN WITH INPUT LAYER BEFORE ADDING MORE LAYERS");
}

vector<double> MultiLayerParatron::run(vector<double> x) {
    outputs[0] = x;
    for (int i = 1;i < network.size();i++) {
        for (int j = 0;j < cells_in_layer[i];j++) {
            outputs[i][j] = network[i][j].run(outputs[i - 1]);
        }
    }
    if (network[network.size() - 1][0].A_F == SOFTMAX) {
        vector<double> last = outputs.back();
        //for (int j = 0;j < cells_in_layer.back();j++) {
        //    double temp = exp(last[j]);
        //    if (isnan(temp))
        //        printf("euler wha\n");
        //    outputs.back()[j] = temp;
        //}
        double denom = accumulate(outputs.back().begin(), outputs.back().end(), 0.0);
        if (denom < DBL_MIN)
            denom = DBL_MIN;
        if (isnan(denom))
            printf("denom wha\n");
        for (int j = 0;j < cells_in_layer.back();j++) {
            double temp = outputs.back()[j] / denom;
            if (isnan(temp))
                printf("softmax wha\n");
            outputs.back()[j] /= denom;
        }
    }
    return outputs.back();
}

vector<double> MultiLayerParatron::getRun(double* d_x) {
    runMLP << <1, 1 >> > (d_x, d_outputs, d_weights, d_A_Fs, d_CIL, cells_in_layer.size(), bias, d_weightLayerOffsets, d_outputLayerOffsets);
    cudaDeviceSynchronize();
    return cudaCopy2dBackToVector(&d_outputs, cells_in_layer).back();
}

vector<vector<double>> MultiLayerParatron::getOut(double* d_x) {

    runMLP << <1, 1 >> > (d_x, d_outputs, d_weights, d_A_Fs, d_CIL, cells_in_layer.size(), bias, d_weightLayerOffsets, d_outputLayerOffsets);
    cudaDeviceSynchronize();
    return cudaCopy2dBackToVector(&d_outputs, cells_in_layer);
}

void MultiLayerParatron::run(double* d_x) {
    runMLP << <1, 1 >> > (d_x, d_outputs, d_weights, d_A_Fs, d_CIL, cells_in_layer.size(), bias, d_weightLayerOffsets, d_outputLayerOffsets);
    cudaDeviceSynchronize();
}

vector<double> MultiLayerParatron::getCleanRun(double* d_x) {
    //printf("\n\n\n");
    int layers = cells_in_layer.size();
    if (cells_in_layer[0] > 511) {
        copyElements << <cells_in_layer[0] / 32, 32 >> > (d_outputs_href[0], d_x, cells_in_layer[0]);
        cudaDeviceSynchronize();
    }
    else {
        copySeqElements << <1, 1 >> > (d_outputs_href[0], d_x, cells_in_layer[0]);
        cudaDeviceSynchronize();
    }
    //vector<double> in(cells_in_layer[0], 0.0);
    //cudaMemcpy(&in[0], d_outputs_href[0], sizeof(double) * cells_in_layer[0], cudaMemcpyDeviceToHost);
    //printArray(&in[0], cells_in_layer[0]);
    for (int i = 1;i < layers;i++) {
        runCleanParatron << < (cells_in_layer[i] / 32) + 1, 32 >> > (d_outputs_href[i-1], d_outputs_href[i], d_weights_href[i - 1], network[i][0].A_F, cells_in_layer[i - 1], cells_in_layer[i], bias);
        cudaDeviceSynchronize();
        //double* temp = new double[cells_in_layer[i]];
        //cudaMemcpy(temp, d_outputs_href[i], sizeof(double) * cells_in_layer[i], cudaMemcpyDeviceToHost);
        //printArray(temp, cells_in_layer[i]);
    }
    if (network[layers-1][0].A_F == SOFTMAX) {
        SoftMaxSeq << <1, 1 >> > (d_outputs_href[layers - 1], cells_in_layer[layers - 1]);
        //double* temp = new double[cells_in_layer[layers-1]];
        //cudaMemcpy(temp, d_outputs_href[layers - 1], sizeof(double) * cells_in_layer[layers - 1], cudaMemcpyDeviceToHost);
        //printf("FINAL\n");
        //printArray(temp, cells_in_layer[layers - 1]);
        cudaDeviceSynchronize();
    }
    vector<double> out(cells_in_layer[layers-1], 0.0);
    cudaMemcpy(&out[0], d_outputs_href[layers - 1], sizeof(double) * cells_in_layer[layers - 1], cudaMemcpyDeviceToHost);
    return out;
}

void MultiLayerParatron::cleanRun(double* d_x) {
    //printf("\n\n\n");
    int layers = cells_in_layer.size();
    if (cells_in_layer[0] > 511) {
        copyElements << <cells_in_layer[0] / 32, 32 >> > (d_outputs_href[0], d_x, cells_in_layer[0]);
        cudaDeviceSynchronize();
    }
    else {
        copySeqElements << <1, 1 >> > (d_outputs_href[0], d_x, cells_in_layer[0]);
        cudaDeviceSynchronize();
    }
    //vector<double> in(cells_in_layer[0], 0.0);
    //cudaMemcpy(&in[0], d_outputs_href[0], sizeof(double) * cells_in_layer[0], cudaMemcpyDeviceToHost);
    //printArray(&in[0], cells_in_layer[0]);
    for (int i = 1;i < layers;i++) {
        runCleanParatron << < (cells_in_layer[i] / 32) + 1, 32 >> > (d_outputs_href[i - 1], d_outputs_href[i], d_weights_href[i - 1], network[i][0].A_F, cells_in_layer[i - 1], cells_in_layer[i], bias);
        cudaDeviceSynchronize();
        //double* temp = new double[cells_in_layer[i]];
        //cudaMemcpy(temp, d_outputs_href[i], sizeof(double) * cells_in_layer[i], cudaMemcpyDeviceToHost);
        //printArray(temp, cells_in_layer[i]);
    }
    if (network[layers - 1][0].A_F == SOFTMAX) {
        SoftMaxSeq << <1, 1 >> > (d_outputs_href[layers - 1], cells_in_layer[layers - 1]);
        //double* temp = new double[cells_in_layer[layers-1]];
        //cudaMemcpy(temp, d_outputs_href[layers - 1], sizeof(double) * cells_in_layer[layers - 1], cudaMemcpyDeviceToHost);
        //printf("FINAL\n");
        //printArray(temp, cells_in_layer[layers - 1]);
        cudaDeviceSynchronize();
    }
}

double MultiLayerParatron::getLoss(vector<double> x, vector<double> y) {
    double loss = 0.0;
    switch (L_F) {
    case(MSE):
        for (int i = 0;i < x.size();i++) {
            loss += pow((x[i] - y[i]), 2);
        }
        loss /= x.size();
        return loss;
        break;
    case(CROSS_ENTROPY):
        for (int i = 0;i < x.size();i++) {
            if (x[i] == 0.0) x[i] = 0.00001;
            loss -= y[i] * (double)log(x[i]);
        }
        return loss;
        break;
    }
}

void MultiLayerParatron::getLoss(double* x, double* y) {
    getLossSeq << <1, 1 >> > (x, y, d_loss, d_L_F, cells_in_layer[cells_in_layer.size()-1]);
}

double MultiLayerParatron::bp(vector<double> x, vector<double> y) {
    //get outputs
    clock_t run_start, run_end;
    run_start = clock();
    vector<double> o = run(x);
    double su = getSum(o);
    if (isnan(o[0]) || (L_F == CROSS_ENTROPY && (su < .9988888 || su > 1.0011111)))
        printf("o wha\n");
    run_end = clock();

    //get loss
    clock_t mse_start, mse_end;
    mse_start = clock();
    double loss = getLoss(o, y);
    mse_end = clock();

    if (isnan(loss))
        printf("l wha\n");

    //output error term = o * (1-o) * 2 * (o - y)
    clock_t getError_start, getError_end;
    getError_start = clock();
    int s = o.size();
    for (int i = 0;i < s;i++) {
        //vector<double> yo = error_terms.back();
        error_terms.back()[i] = o[i] * (1 - o[i]) * 2 * (o[i] - y[i]);
    }
    getError_end = clock();
    if (isnan(error_terms.back()[0]))
        printf("w wha\n");

    //k = j+1
    //i = layer
    //hidden layer terms(ji) = o(j) *(1 - o(j)) * sum(w(j->k)*err(k))
    clock_t propError_start, propError_end;
    propError_start = clock();
    for (int i = error_terms.size() - 2;i >= 0;i--) {
        for (int j = 0;j < error_terms[i].size();j++) {
            double err_sum = 0;
            for (int k = 0;k < cells_in_layer[i + 1];k++) {
                err_sum += h_weights[i][k][j] * error_terms[i + 1][k];
                if (isnan(err_sum))
                    printf("w wha\n");
            }
            error_terms[i][j] = outputs[i][j] * (1 - outputs[i][j]) * err_sum;
        }
    }
    propError_end = clock();

    //update weights
    //weight += learning rate * error term *
    clock_t weights_start, weights_end;
    weights_start = clock();
    for (int i = 1;i < cells_in_layer.size();i++) {
        for (int j = 0;j < cells_in_layer[i];j++) {
            for (int k = 0;k < cells_in_layer[i - 1] + 1;k++) {
                double delta;
                if (k == cells_in_layer[i - 1])
                    delta = eta * error_terms[i][j] * bias;
                else
                    delta = eta * error_terms[i][j] * outputs[i - 1][k];
                h_weights[i-1][j][k] += delta;
                if (isnan(delta))
                    printf("w wha\n");
            }
        }
    }
    weights_end = clock();
    //printExecution("Run NN", run_start, run_end);
    //printExecution("Get MSE", mse_start, mse_end);
    //printExecution("Get Error", getError_start, getError_end);
    //printExecution("Propogate Error", propError_start, propError_end);
    //printExecution("Update Weights", weights_start, weights_end);

    return loss;
}

double MultiLayerParatron::bp(double* x, vector<double> y) {
    //get outputs
    clock_t run_start, run_end;
    run_start = clock();
    vector<double> o = getRun(x);
    double su = getSum(o);
    if (isnan(o[0]) || (L_F == CROSS_ENTROPY && (su < .9988888 || su > 1.0011111)))
        printf("o wha\n");
    run_end = clock();

    //get loss
    clock_t mse_start, mse_end;
    mse_start = clock();
    double loss = getLoss(o, y);
    mse_end = clock();

    if (isnan(loss))
        printf("l wha\n");

    //ACTUALLY, it may be different
    //output error term = o * (1-o) * 2 * (o - y)
    clock_t getError_start, getError_end;
    getError_start = clock();
    int s = o.size();
    for (int i = 0;i < s;i++) {
        //vector<double> yo = error_terms.back();
        error_terms.back()[i] = o[i] * (1 - o[i]) * 2 * (o[i] - y[i]);
    }
    getError_end = clock();
    if (isnan(error_terms.back()[0]))
        printf("w wha\n");

    //k = j+1
    //i = layer
    //hidden layer terms(ji) = o(j) *(1 - o(j)) * sum(w(j->k)*err(k))
    clock_t propError_start, propError_end;
    propError_start = clock();
    for (int i = error_terms.size() - 2;i >= 0;i--) {
        for (int j = 0;j < error_terms[i].size();j++) {
            double err_sum = 0;
            for (int k = 0;k < cells_in_layer[i + 1];k++) {
                err_sum += h_weights[i][k][j] * error_terms[i + 1][k];
                if (isnan(err_sum))
                    printf("w wha\n");
            }
            error_terms[i][j] = outputs[i][j] * (1 - outputs[i][j]) * err_sum;
        }
    }
    propError_end = clock();

    //update weights
    //weight += learning rate * error term *
    clock_t weights_start, weights_end;
    weights_start = clock();
    for (int i = 1;i < cells_in_layer.size();i++) {
        for (int j = 0;j < cells_in_layer[i];j++) {
            for (int k = 0;k < cells_in_layer[i - 1] + 1;k++) {
                double delta;
                if (k == cells_in_layer[i - 1])
                    delta = eta * error_terms[i][j] * bias;
                else
                    delta = eta * error_terms[i][j] * outputs[i - 1][k];
                h_weights[i - 1][j][k] += delta;
                if (isnan(delta))
                    printf("w wha\n");
            }
        }
    }
    weights_end = clock();
    //printExecution("Run NN", run_start, run_end);
    //printExecution("Get MSE", mse_start, mse_end);
    //printExecution("Get Error", getError_start, getError_end);
    //printExecution("Propogate Error", propError_start, propError_end);
    //printExecution("Update Weights", weights_start, weights_end);
    cudaMemcpy3dOffVector(&d_weights, h_weights);
    return loss;
}

double MultiLayerParatron::bp(double* x, double* y) {

    runMLP<<<1,1>>>(x, d_outputs, d_weights, d_A_Fs, d_CIL, cells_in_layer.size(), bias, d_weightLayerOffsets, d_outputLayerOffsets);
    cudaDeviceSynchronize();

    backpropagation << <1, 1 >> > (d_loss, x, y, d_error_terms, d_outputs, d_weights, d_A_Fs, d_CIL, cells_in_layer.size(), bias, eta, d_weightLayerOffsets, d_outputLayerOffsets);
    cudaDeviceSynchronize();

    double* loss = new double;
    //error_terms = cudaCopy2dBackToVector(&d_error_terms, {512, 512, 10});
    cudaMemcpy(loss, d_loss, sizeof(double), cudaMemcpyDeviceToHost);

    //h_weights = cudaCopy3dBackToVector(&d_weights, weight_lengths);

    return *loss;

}

double MultiLayerParatron::cleanbp(double* x, vector<double> y) {
    //get outputs
    clock_t run_start, run_end;
    run_start = clock();
    vector<double> o = getCleanRun(x);
    double su = getSum(o);
    if (isnan(o[0]) || (L_F == CROSS_ENTROPY && (su < .9988888 || su > 1.0011111)))
        printf("o wha\n");
    run_end = clock();

    //get loss
    clock_t mse_start, mse_end;
    mse_start = clock();
    double loss = getLoss(o, y);
    mse_end = clock();

    if (isnan(loss))
        printf("l wha\n");

    //ACTUALLY, it may be different
    //output error term = o * (1-o) * 2 * (o - y)
    clock_t getError_start, getError_end;
    getError_start = clock();
    int s = o.size();
    for (int i = 0;i < s;i++) {
        //vector<double> yo = error_terms.back();
        error_terms.back()[i] = o[i] * (1 - o[i]) * 2 * (o[i] - y[i]);
    }
    getError_end = clock();
    if (isnan(error_terms.back()[0]))
        printf("w wha\n");

    //k = j+1
    //i = layer
    //hidden layer terms(ji) = o(j) *(1 - o(j)) * sum(w(j->k)*err(k))
    clock_t propError_start, propError_end;
    propError_start = clock();
    for (int i = error_terms.size() - 2;i >= 0;i--) {
        for (int j = 0;j < error_terms[i].size();j++) {
            double err_sum = 0;
            for (int k = 0;k < cells_in_layer[i + 1];k++) {
                err_sum += h_weights[i][k][j] * error_terms[i + 1][k];
                if (isnan(err_sum))
                    printf("w wha\n");
            }
            error_terms[i][j] = outputs[i][j] * (1 - outputs[i][j]) * err_sum;
        }
    }
    propError_end = clock();

    //update weights
    //weight += learning rate * error term *
    clock_t weights_start, weights_end;
    weights_start = clock();
    for (int i = 1;i < cells_in_layer.size();i++) {
        for (int j = 0;j < cells_in_layer[i];j++) {
            for (int k = 0;k < cells_in_layer[i - 1] + 1;k++) {
                double delta;
                if (k == cells_in_layer[i - 1])
                    delta = eta * error_terms[i][j] * bias;
                else
                    delta = eta * error_terms[i][j] * outputs[i - 1][k];
                h_weights[i - 1][j][k] += delta;
                if (isnan(delta))
                    printf("w wha\n");
            }
        }
    }
    weights_end = clock();
    //printExecution("Run NN", run_start, run_end);
    //printExecution("Get MSE", mse_start, mse_end);
    //printExecution("Get Error", getError_start, getError_end);
    //printExecution("Propogate Error", propError_start, propError_end);
    //printExecution("Update Weights", weights_start, weights_end);
    cudaMemcpy3dOffVectorHostRef(&d_weights_href, h_weights);
    return loss;
}

double MultiLayerParatron::cleanbp(double* x, double* y) {
    int LL = cells_in_layer.size() - 1;
    //get outputs
    clock_t run_start, run_end;
    run_start = clock();
    cleanRun(x);
    //double su = getSum(o);
    //if (isnan(o[0]) || (L_F == CROSS_ENTROPY && (su < .9988888 || su > 1.0011111)))
    //    printf("o wha\n");
    run_end = clock();

    //get loss
    clock_t mse_start, mse_end;
    mse_start = clock();
    //make gpu-side loss variable
    //getLoss(d_outputs_href[cells_in_layer[cells_in_layer.size()-1]], y);
    mse_end = clock();

    //ACTUALLY, it may be different
    //output error term = o * (1-o) * (y - o)
    clock_t getError_start, getError_end;
    getError_start = clock();
    int s = cells_in_layer[cells_in_layer.size() - 1];

    getErrorLayerWRTInputSeq<<<1,1>>>(d_error_terms_href[LL], d_outputs_href[LL], y, s, L_F, A_Fs[LL]);

    //getError_end = clock();

    //k = j+1
    //i = layer
    //clock_t propError_start, propError_end;
    //propError_start = clock();
    for (int i = error_terms.size() - 2;i >= 0;i--) {
    //    for (int j = 0;j < error_terms[i].size();j++) {
    //        double err_sum = 0;
    //        for (int k = 0;k < cells_in_layer[i + 1];k++) {
    //            err_sum += h_weights[i][k][j] * error_terms[i + 1][k];
    //            if (isnan(err_sum))
    //                printf("w wha\n");
    //        }
    //        error_terms[i][j] = outputs[i][j] * (1 - outputs[i][j]) * err_sum;
    //    }
    }
    //propError_end = clock();

    ////update weights
    //// weight -= learning rate * gradient->(rate or derivative)
    ////weight += learning rate * error term *
    //clock_t weights_start, weights_end;
    //weights_start = clock();
    //for (int i = 1;i < cells_in_layer.size();i++) {
    //    for (int j = 0;j < cells_in_layer[i];j++) {
    //        for (int k = 0;k < cells_in_layer[i - 1] + 1;k++) {
    //            double delta;
    //            if (k == cells_in_layer[i - 1])
    //                delta = eta * error_terms[i][j] * bias;
    //            else
    //                delta = eta * error_terms[i][j] * outputs[i - 1][k];
    //            h_weights[i - 1][j][k] += delta;
    //            if (isnan(delta))
    //                printf("w wha\n");
    //        }
    //    }
    //}
    //weights_end = clock();
    ////printExecution("Run NN", run_start, run_end);
    ////printExecution("Get MSE", mse_start, mse_end);
    ////printExecution("Get Error", getError_start, getError_end);
    ////printExecution("Propogate Error", propError_start, propError_end);
    ////printExecution("Update Weights", weights_start, weights_end);
    //cudaMemcpy3dOffVectorHostRef(&d_weights_href, h_weights);
    //return loss;
    return 0.0;
}