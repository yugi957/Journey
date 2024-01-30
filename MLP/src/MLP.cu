#include "../Headers/MLP.cuh"
#include "../../CUDA/Headers/kernels.cuh"

//MLP class methods:
MultiLayerParatron::MultiLayerParatron(vector<int> CIL, loss_function func, double bias, double eta, double momentum, int batchSize) {
    this->cells_in_layer = CIL;
    this->L_F = func;
    this->bias = bias;
    this->eta = eta;
    this->batchSize = batchSize;
    this->momentum = momentum;

    for (int i = 0;i < cells_in_layer.size();i++) {
        outputs.push_back(vector<double>(cells_in_layer[i], 0.0));
        //error_terms.push_back(vector<double>(cells_in_layer[i], 0.0));
        //clock_t gpu_start, gpu_end;
        if (i > 0) { //input layer has no neurons
            for (int j = 0;j < cells_in_layer[i];j++) {
                //gpu_start = clock();
                //gpu_end = clock();
                //printf("%d : %d ", cells_in_layer[i], j);
                //printExecution("Loop", gpu_start, gpu_end);
            }
        }
    }
}

void MultiLayerParatron::finalize() {
    weightLayerOffsets = new int[cells_in_layer.size() - 1];
    weightLayerOffsets[0] = 0;
    for (int i = 1;i < cells_in_layer.size() - 1;i++) {
        weightLayerOffsets[i] = weightLayerOffsets[i - 1] + (cells_in_layer[i] * (cells_in_layer[i - 1] + 1));
    }
    cudaMalloc((void**)&d_weightLayerOffsets, sizeof(int) * (cells_in_layer.size() - 1));
    cudaMemcpy(d_weightLayerOffsets, weightLayerOffsets, sizeof(int) * (cells_in_layer.size() - 1), cudaMemcpyHostToDevice);

    outputLayerOffsets = new int[cells_in_layer.size()];
    outputLayerOffsets[0] = 0;
    for (int i = 1;i <= cells_in_layer.size();i++) {
        outputLayerOffsets[i] = outputLayerOffsets[i - 1] + cells_in_layer[i - 1];
    }
    cudaMalloc((void**)&d_outputLayerOffsets, sizeof(int) * (cells_in_layer.size() + 1));
    cudaMemcpy(d_outputLayerOffsets, outputLayerOffsets, sizeof(int) * (cells_in_layer.size() + 1), cudaMemcpyHostToDevice);
    int size = sizeof(double);
    cudaMalloc((void**)&d_CIL, size * cells_in_layer.size());
    cudaMemcpy(d_CIL, &cells_in_layer[0], size * cells_in_layer.size(), cudaMemcpyHostToDevice);
    cudaAllocate2dOffVector(&d_outputs, outputs);
    cudaAllocate2dOffVectorHostRef(&d_outputs_href, outputs);
    for (int i = 0;i < h_weights.size();i++) {
        xavier_init(h_weights[i], h_weights[i][0].size(), h_weights[i].size());
    }
    for (int i = 0;i < h_weights.size();i++) {
        gradient.push_back(vector<vector<double>>());
        for (int j = 0;j < h_weights[i].size();j++) {
            gradient[i].push_back(vector<double>(h_weights[i][j].size(), 0));
        }
    }
    cudaAllocate3dOffVector(&d_weights, h_weights);
    for (int i = 1;i < cells_in_layer.size();i++) {
        weight_lengths.push_back(vector<int>());
        for (int j = 0;j < cells_in_layer[i];j++) {
            weight_lengths[i - 1].push_back(cells_in_layer[i - 1] + 1);
        }
    }
    cudaAllocate3dOffVectorHostRef(&d_weights_href, h_weights);
    cudaAllocate3dOffVectorHostRef(&d_gradient_href, gradient);

    cudaMalloc((void**)&d_A_Fs, sizeof(activation_function) * h_A_Fs.size());
    cudaMemcpy(d_A_Fs, &h_A_Fs[0], sizeof(activation_function) * h_A_Fs.size(), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_L_F, sizeof(loss_function));
    cudaMemcpy(d_L_F, &L_F, sizeof(loss_function), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_loss, sizeof(double));
    cudaMalloc((void**)&d_eta, sizeof(double));
    cudaMemcpy(d_eta, &eta, sizeof(double), cudaMemcpyHostToDevice);
    cudaAllocate2dOffVector(&d_error_terms, error_terms);
    cudaAllocate2dOffVectorHostRef(&d_error_terms_href, error_terms);

    //batch allocation
    if (batchSize > 0) {
        batch_out = vector<vector<double>>(outputs.size());
        batch_err = vector<vector<double>>(error_terms.size());
        batch_gradient = vector<vector<double>>(h_weights.size());
        for (int i = 0;i < outputs.size();i++) batch_out[i] = (vector<double>(outputs[i].size() * batchSize));
        for (int i = 0;i < error_terms.size();i++) batch_err[i] = (vector<double>(error_terms[i].size() * batchSize));
        for (int i = 0;i < h_weights.size();i++) batch_gradient[i] = (vector<double>(h_weights[i].size() * h_weights[i][0].size() * batchSize));
        cudaAllocate2dOffVectorHostRef(&d_batch_outs_href, batch_out);
        cudaAllocate2dOffVectorHostRef(&d_batch_errors_href, batch_err);
        cudaAllocate2dOffVectorHostRef(&d_batch_grad_href, batch_gradient);
    }



    //cudaCopy3dBackToVector(&d_weights, weight_lengths);

}

void MultiLayerParatron::addLayer(int CIL, activation_function func) {
    if (cells_in_layer.size() > 0) { //means input layer is done
        int LL = cells_in_layer.size() - 1; //Last Layer index before new layer
        cells_in_layer.push_back(CIL);
        outputs.push_back(vector<double>(CIL, 0.0));
        error_terms.push_back(vector<double>(CIL, 0.0));
        h_weights.push_back(vector<vector<double>>(CIL, vector<double>(cells_in_layer[LL] + 1, 0.0)));
        h_A_Fs.push_back(func);
        for (int i = 0;i < CIL;i++) {
            generate(h_weights[LL][i].begin(), h_weights[LL][i].end(), frand);
        }
    }
    else printf("INITIALIZE NN WITH INPUT LAYER BEFORE ADDING MORE LAYERS");
}

void MultiLayerParatron::cleanerRun(double* d_x) {
    int layers = cells_in_layer.size();
    if (cells_in_layer[0] > 511) {
        copyElements << <cells_in_layer[0] / 32 + 1, 32 >> > (d_outputs_href[0], d_x, cells_in_layer[0]);
        cudaDeviceSynchronize();
    }
    else {
        copySeqElements << <1, 1 >> > (d_outputs_href[0], d_x, cells_in_layer[0]);
        cudaDeviceSynchronize();
    }
    for (int i = 1;i < layers;i++) {
        runCleanParatron << < (cells_in_layer[i] / 32) + 1, 32 >> > (d_outputs_href[i - 1], d_outputs_href[i], d_weights_href[i - 1], h_A_Fs[i - 1], cells_in_layer[i - 1], cells_in_layer[i], bias);
        cudaDeviceSynchronize();
    }
    if (h_A_Fs[layers - 2] == SOFTMAX) {
        SoftMaxSeq << <1, 1 >> > (d_outputs_href[layers - 1], cells_in_layer[layers - 1]);
        cudaDeviceSynchronize();
    }
}

vector<double> MultiLayerParatron::getRun(double* d_x) {
    int layers = cells_in_layer.size();
    cleanerRun(d_x);
    vector<double> out(cells_in_layer[layers - 1], 0.0);
    cudaMemcpy(&out[0], d_outputs_href[layers - 1], sizeof(double) * cells_in_layer[layers - 1], cudaMemcpyDeviceToHost);
    return out;
}

void MultiLayerParatron::batchRun(double* d_batchX) {
    int layers = cells_in_layer.size();
    if (cells_in_layer[0] > 511) {
        batchCopy << <(cells_in_layer[0] * batchSize / 32) + 1, 32 >> > (d_batch_outs_href[0], d_batchX, cells_in_layer[0] * batchSize);
        gpuErrorchk(cudaDeviceSynchronize());
    }
    else {
        printf("BATCH SIZE * INPUT SIZE IS TOO LOW\n");
        copySeqElements << <1, 1 >> > (d_batch_outs_href[0], d_batchX, cells_in_layer[0] * batchSize);
        gpuErrorchk(cudaDeviceSynchronize());
    }
    for (int i = 1;i < layers;i++) {
        runBatchParatron << < dim3(batchSize, (cells_in_layer[i] / 32) + 1), 32 >> > (d_batch_outs_href[i - 1], d_batch_outs_href[i], d_weights_href[i - 1], h_A_Fs[i - 1], cells_in_layer[i - 1], cells_in_layer[i] * batchSize, cells_in_layer[i], bias);
        gpuErrorchk(cudaDeviceSynchronize());
    }
    if (h_A_Fs[layers - 2] == SOFTMAX) {
        batchSoftMax << <1, batchSize >> > (d_batch_outs_href[layers - 1], cells_in_layer[layers - 1], batchSize);
        gpuErrorchk(cudaDeviceSynchronize());
    }
}

vector<vector<double>> MultiLayerParatron::getBatchRun(double* d_batchX) {
    int layers = cells_in_layer.size();
    batchRun(d_batchX);
    return cudaCopyBatchBackToVectorHref(&d_batch_outs_href[layers - 1], cells_in_layer[layers - 1], batchSize);
}

void MultiLayerParatron::getLoss(double* x, double* y) {
    getLossSeq << <1, 1 >> > (x, y, d_loss, d_L_F, cells_in_layer[cells_in_layer.size() - 1]);
    cudaDeviceSynchronize();
}

void MultiLayerParatron::bLoss(double* x, double* y) {
    batchLoss << <1, 1 >> > (x, y, d_loss, CROSS_ENTROPY, cells_in_layer[cells_in_layer.size() - 1], batchSize);
    gpuErrorchk(cudaDeviceSynchronize());
}

double MultiLayerParatron::cleanerbp(double* x, double* y) {
    //get outputs
    cleanerRun(x);

    //get loss
    //make gpu-side loss variable
    getLoss(d_outputs_href[cells_in_layer.size() - 1], y);

    //output error term = o * (1-o) * (y - o)
    int s = cells_in_layer[cells_in_layer.size() - 1];

    getErrorLayerWRTInputSeq << <1, 1 >> > (d_error_terms_href[cells_in_layer.size() - 2], d_outputs_href[cells_in_layer.size() - 1], y, s, L_F, SOFTMAX);
    cudaDeviceSynchronize();

    for (int i = cells_in_layer.size() - 3;i >= 0;i--) {
        cleanGradient << <(cells_in_layer[i + 1] / 32) + 1, 32 >> > (d_weights_href[i], d_error_terms_href[i + 1], d_error_terms_href[i], d_outputs_href[i + 1], cells_in_layer[i + 2], cells_in_layer[i + 1]);
        cudaDeviceSynchronize();
    }

    for (int i = 0;i < cells_in_layer.size() - 1;i++) {
        cleanUpdateWeightsbyLayer << <cells_in_layer[i] + 1, cells_in_layer[i + 1] >> > (d_weights_href[i], d_error_terms_href[i], d_outputs_href[i], eta, cells_in_layer[i], cells_in_layer[i + 1], bias);
        cudaDeviceSynchronize();
    }

    double* loss = new double;
    cudaMemcpy(loss, d_loss, sizeof(double), cudaMemcpyDeviceToHost);
    //this->h_weights = cudaCopy3dBackToVectorHref(&d_weights_href, weight_lengths);
    return *loss;
}

vector<vector<double>> MultiLayerParatron::getCleanerBp(double* x, double* y) {
    int layers = cells_in_layer.size();
    cleanerbp(x, y);
    //return cudaCopy2dBackToVectorHref(d_error_terms_href, vector<int>({ 512,512,10 }));
    //return cudaCopy2dBackToVectorHref(&d_error_terms_href[layers - 1], vector<int>({10}));
    this->h_weights = cudaCopy3dBackToVectorHref(&d_weights_href, weight_lengths);
    return cudaCopy2dBackToVectorHref(d_weights_href, vector<int>({ 785 * 512,513 * 512,513 * 10 }));
}

double MultiLayerParatron::batchP(double* batchX, double* batchY) {

    //get outputs
    batchRun(batchX);

    //get loss
    //make gpu-side loss variable
    bLoss(d_batch_outs_href[cells_in_layer.size() - 1], batchY);

    //output error term = o * (1-o) * (y - o)
    int s = cells_in_layer[cells_in_layer.size() - 1];

    batchErrorLayer << <1, batchSize >> > (d_batch_errors_href[cells_in_layer.size() - 2], d_batch_outs_href[cells_in_layer.size() - 1], batchY, s, L_F, SOFTMAX);
    gpuErrorchk(cudaDeviceSynchronize());

    for (int i = cells_in_layer.size() - 3;i >= 0;i--) {
        batchGradient << <dim3(batchSize, (cells_in_layer[i + 1] / 32) + 1), 32 >> > (d_weights_href[i], d_batch_errors_href[i + 1], d_batch_errors_href[i], d_batch_outs_href[i + 1], cells_in_layer[i + 2], cells_in_layer[i + 1], cells_in_layer[i + 1] * batchSize);
        gpuErrorchk(cudaDeviceSynchronize());
    }

    //for (int i = 0;i < cells_in_layer.size() - 1;i++) {
    //    batchMakeGradient << < dim3(batchSize, cells_in_layer[i] + 1), cells_in_layer[i + 1] >> > (d_batch_grad_href[i], d_batch_errors_href[i], d_batch_outs_href[i], eta, cells_in_layer[i], cells_in_layer[i + 1], bias);
    //    gpuErrorchk(cudaDeviceSynchronize());
    //}

    //for (int i = 0;i < h_weights.size();i++) {
    //    averageGrad << <cells_in_layer[i] + 1, cells_in_layer[i + 1] >> > (d_batch_grad_href[i], d_gradient_href[i], batchSize, h_weights[i].size(), h_weights[i][0].size());
    //}
    //cudaDeviceSynchronize();
    //Average or try lower learning rater

    for (int b = 0;b < batchSize;b++) {
        for (int i = 0;i < cells_in_layer.size() - 1;i++) {
            batchUpdateWeightsbyLayer << <cells_in_layer[i] + 1, cells_in_layer[i + 1] >> > (d_weights_href[i], d_batch_errors_href[i], d_batch_outs_href[i], eta, cells_in_layer[i], cells_in_layer[i + 1], bias, b);
            gpuErrorchk(cudaDeviceSynchronize());
        }
    }

    //for (int i = 0;i < h_weights.size();i++) {
    //    applyGrad << <cells_in_layer[i] + 1, cells_in_layer[i + 1] >> > (d_weights_href[i], d_gradient_href[i], h_weights[i].size());
    //}
    //gpuErrorchk(cudaDeviceSynchronize());

    double* loss = new double;
    cudaMemcpy(loss, d_loss, sizeof(double), cudaMemcpyDeviceToHost);
    //this->h_weights = cudaCopy3dBackToVectorHref(&d_weights_href, weight_lengths);

    return *loss;
}

vector<vector<double>> MultiLayerParatron::getBatchP(double* batchX, double* batchY) {
    int layers = cells_in_layer.size();
    batchP(batchX, batchY);
    return cudaCopy2dBackToVectorHref(d_batch_errors_href, vector<int>({ 512,512,10 }));
    //return cudaCopyBatchBackToVectorHref(&d_batch_outs_href[layers - 1], cells_in_layer[layers - 1], batchSize);
    //return cudaCopy2dBackToVectorHref(d_weights_href, vector<int>({ 785*512,513*512,513*10 }));
}

double MultiLayerParatron::aveBatchP(double* batchX, double* batchY) {

    //get outputs
    batchRun(batchX);

    //get loss
    //make gpu-side loss variable
    bLoss(d_batch_outs_href[cells_in_layer.size() - 1], batchY);

    //output error term = o * (1-o) * (y - o)
    int s = cells_in_layer[cells_in_layer.size() - 1];

    batchErrorLayer << <1, batchSize >> > (d_batch_errors_href[cells_in_layer.size() - 2], d_batch_outs_href[cells_in_layer.size() - 1], batchY, s, L_F, SOFTMAX);
    gpuErrorchk(cudaDeviceSynchronize());

    for (int i = cells_in_layer.size() - 3;i >= 0;i--) {
        batchGradient << <dim3(batchSize, (cells_in_layer[i + 1] / 32) + 1), 32 >> > (d_weights_href[i], d_batch_errors_href[i + 1], d_batch_errors_href[i], d_batch_outs_href[i + 1], cells_in_layer[i + 2], cells_in_layer[i + 1], cells_in_layer[i + 1] * batchSize);
        gpuErrorchk(cudaDeviceSynchronize());
    }

    for (int i = 0;i < cells_in_layer.size() - 1;i++) {
        batchMakeGradient << < dim3(batchSize, cells_in_layer[i] + 1), cells_in_layer[i + 1] >> > (d_batch_grad_href[i], d_gradient_href[i], d_batch_errors_href[i], d_batch_outs_href[i], eta, momentum, cells_in_layer[i], cells_in_layer[i + 1], bias, batchSize);
        gpuErrorchk(cudaDeviceSynchronize());
    }

    for (int i = 0;i < h_weights.size();i++) {
        averageGrad << <cells_in_layer[i] + 1, cells_in_layer[i + 1] >> > (d_batch_grad_href[i], d_gradient_href[i], batchSize, (cells_in_layer[i] + 1), cells_in_layer[i + 1]);
    }
    gpuErrorchk(cudaDeviceSynchronize());

    for (int i = 0;i < h_weights.size();i++) {
        applyGrad << <cells_in_layer[i] + 1, cells_in_layer[i + 1] >> > (d_weights_href[i], d_gradient_href[i], (cells_in_layer[i] + 1) * cells_in_layer[i + 1]);
    }
    gpuErrorchk(cudaDeviceSynchronize());

    double* loss = new double;
    cudaMemcpy(loss, d_loss, sizeof(double), cudaMemcpyDeviceToHost);
    //this->h_weights = cudaCopy3dBackToVectorHref(&d_weights_href, weight_lengths);

    return *loss;
}

vector<vector<double>> MultiLayerParatron::getAveP(double* batchX, double* batchY) {
    int layers = cells_in_layer.size();
    aveBatchP(batchX, batchY);
    //return cudaCopy2dBackToVectorHref(d_batch_errors_href, vector<int>({ 512,512,10 }));
    //return cudaCopyBatchBackToVectorHref(&d_batch_outs_href[layers - 1], cells_in_layer[layers - 1], batchSize);
    //printf("CHECK WEIGHTS: \n\n");
    //compare3D(this->h_weights, cudaCopy3dBackToVectorHref(&d_weights_href, weight_lengths));
    double* test = new double;
    //cudaMemcpy(test, d_weights_href[0], sizeof(double), cudaMemcpyDeviceToHost);
    //printf("WEIGHT TEST: %f\n", test);
    return cudaCopy2dBackToVectorHref(d_weights_href, vector<int>({ 785 * 512,513 * 512,513 * 10 }));
}