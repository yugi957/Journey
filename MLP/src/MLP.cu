#include "../Headers/MLP.cuh"
#include "../../CUDA/Headers/kernels.cuh"
#include "../../IP/Headers/kernels.cuh"

//MLP class methods:
MultiLayerParatron::MultiLayerParatron(vector<int> CIL, loss_function func, float bias, float eta, float momentum, int batchSize) {
    this->cells_in_layer = CIL;
    this->L_F = func;
    this->bias = bias;
    this->eta = eta;
    this->batchSize = batchSize;
    this->momentum = momentum;

    for (int i = 0;i < cells_in_layer.size();i++) {
        outputs.push_back(vector<float>(cells_in_layer[i], 0.0));
        if (i == 0) { //input layer has no neurons
            layers.push_back(INPUT);
            dimensions.push_back(vec3(CIL[0], 1, 1));
        }
    }
}

void MultiLayerParatron::finalize() {
    int size = sizeof(float);
    cudaAllocate2dOffVectorHostRef(&d_outputs_href, outputs);
    for (int i = 0;i < h_weights.size();i++) {
        if (h_weights[i].size() > 0)
            he_init(h_weights[i], h_weights[i][0].size(), h_weights[i].size());
    }
    for (int i = 0;i < h_weights.size();i++) {
        gradient.push_back(vector<vector<float>>());
        moments.push_back(vector<vector<float>>());
        for (int j = 0;j < h_weights[i].size();j++) {
            gradient[i].push_back(vector<float>(h_weights[i][j].size(), 0));
            moments[i].push_back(vector<float>(h_weights[i][j].size(), 0));
        }
    }
    for (int i = 1;i < cells_in_layer.size();i++) {
        weight_lengths.push_back(vector<int>());
        for (int j = 0;j < cells_in_layer[i];j++) {
            weight_lengths[i - 1].push_back(cells_in_layer[i - 1] + 1);
        }
    }
    cudaAllocate3dOffVectorHostRef(&d_weights_href, h_weights);
    cudaAllocate3dOffVectorHostRef(&d_gradient_href, gradient);
    cudaAllocate3dOffVectorHostRef(&d_moments_href, moments);

    cudaMalloc((void**)&d_A_Fs, sizeof(activation_function) * A_Fs.size());
    cudaMemcpy(d_A_Fs, &A_Fs[0], sizeof(activation_function) * A_Fs.size(), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_L_F, sizeof(loss_function));
    cudaMemcpy(d_L_F, &L_F, sizeof(loss_function), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_loss, sizeof(float));
    cudaMalloc((void**)&d_eta, sizeof(float));
    cudaMemcpy(d_eta, &eta, sizeof(float), cudaMemcpyHostToDevice);
    cudaAllocate2dOffVectorHostRef(&d_error_terms_href, error_terms);

    for (int i = 0;i < conv_weights.size();i++) {
        conv_gradient.push_back(vector<vector<vector<vector<float>>>>());
        for (int j = 0;j < conv_weights[i].size();j++) {
            conv_gradient[i].push_back(vector<vector<vector<float>>>());
            for (int k = 0;k < conv_weights[i][j].size();k++) {
                conv_gradient[i][j].push_back(vector<vector<float>>());
                for (int l = 0;l < conv_weights[i][j][k].size();l++) {
                    conv_gradient[i][j][k].push_back(vector<float>());
                    for (int m = 0;m < conv_weights[i][j][k][l].size();m++) {
                        conv_gradient[i][j][k][l].push_back(0.0);
                    }
                }
            }
        }
    }
    for (int i = 0;i < conv_weights.size();i++) {
        if (layers[i + 1] == DENSE) {
            he_init(conv_weights[i][0][0], conv_weights[i][0][0][0].size(), conv_weights[i][0][0].size());
        }
        if (layers[i + 1] == CONV)
            he_init_conv(conv_weights[i], conv_weights[i][0][0][0].size(), conv_weights[i].size(), conv_weights[i][0].size());
    }
    cout << "------ Summary ------" << endl;
    cout << "input layer: " << dimensions[0].x << "x" << dimensions[0].y << "x" << dimensions[0].z << endl;
    int c = 0;
    for (int i = 1;i < layers.size();i++) {
        if (dimensions[0].y == 1) cout << "linear layer " << i << ": " << dimensions[i].x << "x" << dimensions[i].y << "x" << dimensions[i].z << endl;
        else {
            switch (layers[i]) {
            case(CONV):
                cout << "CONV Layer: ";
                cout << dimensions[i - 1].x << "x" << dimensions[i - 1].y << "x" << dimensions[i - 1].z << " by " <<
                    conv_weights[i - 1][0].size() << "x" << conv_weights[i - 1][0].size() << "x" << conv_weights[i - 1][0][0][0].size() <<
                    " with padding: " << conv_params[c][0] << " and stride: " << conv_params[c][1] << ", " <<
                    conv_weights[c].size() << "times\n";
                break;
            case(MAX_POOL):
                cout << "MAX POOL Layer: \n";
                break;
            }
            cout << "output of layer " << i << ": " << dimensions[i].x << "x" << dimensions[i].y << "x" << dimensions[i].z << endl;
            c++;
        }
    }
    vector<vector<float>> convs2d;
    vector<vector<float>> conv_grads2d;
    for (int i = 0;i < conv_weights.size();i++) {
        convs2d.push_back(flatten4D(conv_weights[i]));
        conv_grads2d.push_back(flatten4D(conv_gradient[i]));
    }
    cudaAllocate2dOffVectorHostRef(&d_conv_weights_href, convs2d);
    cudaAllocate2dOffVectorHostRef(&d_conv_gradient_href, conv_grads2d);

    //batch allocation
    if (batchSize > 0) {
        streams = new cudaStream_t[h_weights.size()];
        for (int i = 0;i < h_weights.size();i++) cudaStreamCreate(&streams[i]);
        batch_out = vector<vector<float>>(outputs.size());
        batch_err = vector<vector<float>>(error_terms.size());
        batch_gradient = vector<vector<float>>(h_weights.size());
        batch_moments = vector<vector<float>>(h_weights.size());
        for (int i = 0;i < outputs.size();i++) batch_out[i] = (vector<float>(outputs[i].size() * batchSize));
        for (int i = 0;i < error_terms.size();i++) batch_err[i] = (vector<float>(error_terms[i].size() * batchSize));
        for (int i = 0;i < h_weights.size();i++) batch_gradient[i] = (vector<float>(h_weights[i].size() * h_weights[i][0].size() * batchSize));
        for (int i = 0;i < h_weights.size();i++) batch_moments[i] = (vector<float>(h_weights[i].size() * h_weights[i][0].size() * batchSize));
        cudaAllocate2dOffVectorHostRef(&d_batch_outs_href, batch_out);
        cudaAllocate2dOffVectorHostRef(&d_batch_errors_href, batch_err);
        cudaAllocate2dOffVectorHostRef(&d_batch_grad_href, batch_gradient);
        cudaAllocate2dOffVectorHostRef(&d_batch_moments_href, batch_moments);
    }



    //cudaCopy3dBackToVector(&d_weights, weight_lengths);

}

//void MultiLayerParatron::addLayer(int CIL, activation_function func) {
//    if (cells_in_layer.size() > 0) { //means input layer is done
//        int LL = cells_in_layer.size() - 1; //Last Layer index before new layer
//        cells_in_layer.push_back(CIL);
//        outputs.push_back(vector<float>(CIL, 0.0));
//        error_terms.push_back(vector<float>(CIL, 0.0));
//        h_weights.push_back(vector<vector<float>>(CIL, vector<float>(cells_in_layer[LL] + 1, 0.0)));
//        A_Fs.push_back(func);
//        for (int i = 0;i < CIL;i++) {
//            generate(h_weights[LL][i].begin(), h_weights[LL][i].end(), frand);
//        }
//    }
//    else printf("INITIALIZE NN WITH INPUT LAYER BEFORE ADDING MORE LAYERS");
//}

void MultiLayerParatron::cleanerRun(float* d_x) {
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
        runCleanParatron << < (cells_in_layer[i] / 32) + 1, 32 >> > (d_outputs_href[i - 1], d_outputs_href[i], d_weights_href[i - 1], A_Fs[i - 1], cells_in_layer[i - 1], cells_in_layer[i], bias);
        cudaDeviceSynchronize();
    }
    if (A_Fs[layers - 2] == SOFTMAX) {
        SoftMaxSeq << <1, 1 >> > (d_outputs_href[layers - 1], cells_in_layer[layers - 1]);
        cudaDeviceSynchronize();
    }
}

void MultiLayerParatron::forward_conv(float* d_x) {
    int num_layers = cells_in_layer.size();
    dim3 gridDim;
    if (cells_in_layer[0] > 511) {
        copyElements << <cells_in_layer[0] / 32 + 1, 32 >> > (d_outputs_href[0], d_x, cells_in_layer[0]);
        cudaDeviceSynchronize();
    }
    else {
        copySeqElements << <1, 1 >> > (d_outputs_href[0], d_x, cells_in_layer[0]);
        cudaDeviceSynchronize();
    }
    for (int i = 1;i < num_layers;i++) {
        int size = dimensions[i].x * dimensions[i].y * dimensions[i].z;
        switch (layers[i]) {
        case(DENSE):
            runCleanParatron << < (cells_in_layer[i] / 32) + 1, 32 >> > (d_outputs_href[i - 1], d_outputs_href[i], d_conv_weights_href[i - 1], A_Fs[i - 1], cells_in_layer[i - 1], cells_in_layer[i], bias);
            cudaDeviceSynchronize();
            break;
        case(CONV):
            gridDim = dim3(dimensions[i].x, dimensions[i].y, (dimensions[i].z + 32 - 1) / 32);
            //gpu_convolve_volume << <gridDim, 32 >> > (dimensions[i - 1], d_outputs_href[i - 1], conv_weights[i-1][0].size(), dimensions[i].z, d_conv_weights_href[i - 1], d_outputs_href[i], conv_params[i - 1][0], conv_params[i - 1][1]);
            size_convolve_volume << <(size + 32 - 1) / 32, 32 >> > (dimensions[i - 1], d_outputs_href[i - 1], conv_weights[i - 1][0].size(), dimensions[i], d_conv_weights_href[i - 1], d_outputs_href[i], conv_params[i - 1][0], conv_params[i - 1][1]);
            cudaDeviceSynchronize();
            //convolve_volume(dimensions[i - 1], outputs[i - 1], vec3(conv_weights[i - 1][0].size(), conv_weights[i - 1][0][0].size(),
                //conv_weights[i - 1][0][0][0].size()), conv_weights[i - 1], dimensions[i], outputs[i],
                //conv_params[i - 1][0], conv_params[i - 1][1]);
            gpu_activate << <(size + 32 - 1) / 32, 32 >> > (d_outputs_href[i], A_Fs[i - 1], size);
            cudaDeviceSynchronize();
            //activate_conv(outputs[i], A_Fs[i - 1]);
            break;
        case(MAX_POOL):
            gridDim = dim3(dimensions[i].x, dimensions[i].y, (dimensions[i].z + 32 - 1) / 32);
            gpu_max_pool << <gridDim, 32 >> > (dimensions[i - 1], d_outputs_href[i - 1], size_t(conv_weights[i - 1][0][0][0][0]), d_outputs_href[i], conv_params[i - 1][0], conv_params[i - 1][1]);
            cudaDeviceSynchronize();
            //max_pool(dimensions[i - 1], outputs[i - 1], dimensions[i], outputs[i], size_t(conv_weights[i - 1][0][0][0][0]), conv_params[i - 1][0], conv_params[i - 1][1]);
            break;
        }
    }
    if (A_Fs[num_layers - 2] == SOFTMAX) {
        SoftMaxSeq << <1, 1 >> > (d_outputs_href[num_layers - 1], cells_in_layer[num_layers - 1]);
        cudaDeviceSynchronize();
    }
}

vector<float> MultiLayerParatron::getForwardConv(float* d_x) {
    int layers = cells_in_layer.size();
    forward_conv(d_x);
    vector<float> out(cells_in_layer[layers - 1], 0.0);
    cudaMemcpy(&out[0], d_outputs_href[layers - 1], sizeof(float) * cells_in_layer[layers - 1], cudaMemcpyDeviceToHost);
    outputs = cudaCopy2dBackToVectorHref(&d_outputs_href, cells_in_layer);
    return out;
}

vector<float> MultiLayerParatron::getRun(float* d_x) {
    int layers = cells_in_layer.size();
    cleanerRun(d_x);
    vector<float> out(cells_in_layer[layers - 1], 0.0);
    cudaMemcpy(&out[0], d_outputs_href[layers - 1], sizeof(float) * cells_in_layer[layers - 1], cudaMemcpyDeviceToHost);
    return out;
}

void MultiLayerParatron::batchRun(float* d_batchX) {
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
        runBatchParatron << < dim3(batchSize, (cells_in_layer[i] / 32) + 1), 32 >> > (d_batch_outs_href[i - 1], d_batch_outs_href[i], d_weights_href[i - 1], A_Fs[i - 1], cells_in_layer[i - 1], cells_in_layer[i] * batchSize, cells_in_layer[i], bias);
        gpuErrorchk(cudaDeviceSynchronize());
    }
    if (A_Fs[layers - 2] == SOFTMAX) {
        batchSoftMax << <1, batchSize >> > (d_batch_outs_href[layers - 1], cells_in_layer[layers - 1], batchSize);
        gpuErrorchk(cudaDeviceSynchronize());
    }
}

vector<vector<float>> MultiLayerParatron::getBatchRun(float* d_batchX) {
    int layers = cells_in_layer.size();
    batchRun(d_batchX);
    return cudaCopyBatchBackToVectorHref(&d_batch_outs_href[layers - 1], cells_in_layer[layers - 1], batchSize);
}

void MultiLayerParatron::getLoss(float* x, float* y) {
    getLossSeq << <1, 1 >> > (x, y, d_loss, d_L_F, cells_in_layer[cells_in_layer.size() - 1]);
    cudaDeviceSynchronize();
}

void MultiLayerParatron::bLoss(float* x, float* y) {
    batchLoss << <1, 1 >> > (x, y, d_loss, CROSS_ENTROPY, cells_in_layer[cells_in_layer.size() - 1], batchSize);
    gpuErrorchk(cudaDeviceSynchronize());
}

float MultiLayerParatron::cleanerbp(float* x, float* y) {
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
        cleanGradient << <(cells_in_layer[i + 1] / 32) + 1, 32 >> > (d_weights_href[i], d_error_terms_href[i + 1], d_error_terms_href[i], d_outputs_href[i + 1], cells_in_layer[i + 2], cells_in_layer[i + 1], A_Fs[i]);
        cudaDeviceSynchronize();
    }

    for (int i = 0;i < cells_in_layer.size() - 1;i++) {
        cleanUpdateWeightsbyLayer << <cells_in_layer[i] + 1, cells_in_layer[i + 1] >> > (d_weights_href[i], d_error_terms_href[i], d_outputs_href[i], eta, cells_in_layer[i], cells_in_layer[i + 1], bias);
        cudaDeviceSynchronize();
    }

    float* loss = new float;
    cudaMemcpy(loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
    //this->h_weights = cudaCopy3dBackToVectorHref(&d_weights_href, weight_lengths);
    return *loss;
}

float MultiLayerParatron::backward_conv(float* d_x, float* d_y) {
    //get outputs
    forward_conv(d_x);

    //get loss
    //make gpu-side loss variable
    getLoss(d_outputs_href[cells_in_layer.size() - 1], d_y);

    //output error term = o * (1-o) * (y - o)
    int s = cells_in_layer[cells_in_layer.size() - 1];

    getErrorLayerWRTInputSeq << <1, 1 >> > (d_error_terms_href[cells_in_layer.size() - 2], d_outputs_href[cells_in_layer.size() - 1], d_y, s, L_F, SOFTMAX);
    cudaDeviceSynchronize();

    for (int i = cells_in_layer.size() - 3;i >= 0;i--) {
        if (layers[i + 2] == CONV) {
            vec3 in_dim = dimensions[i + 1];
            vec3 out_dim = dimensions[i + 2];
            size_t padding = conv_params[i + 1][0];
            size_t stride = conv_params[i + 1][1];
            vec3 k_dim = vec3(conv_weights[i + 1][0].size(), conv_weights[i + 1][0][0].size(), conv_weights[i + 1][0][0][0].size());
            size_t channels = in_dim.z;
            size_t size = in_dim.x * in_dim.y * in_dim.z;
            gpu_backVolve <<<((size + 32) - 1) / 32, 32 >> >(d_outputs_href[i + 1], in_dim, d_error_terms_href[i], k_dim.x, d_conv_weights_href[i + 1], out_dim, d_error_terms_href[i + 1], padding, stride, A_Fs[i]);
            cudaDeviceSynchronize();
        }
        else if (layers[i + 2] == MAX_POOL) {
            vec3 in_dim = dimensions[i + 1];
            vec3 out_dim = dimensions[i + 2];
            size_t pool_size = conv_weights[i + 1][0][0][0][0];
            size_t padding = conv_params[i + 1][0];
            size_t stride = conv_params[i + 1][1];
            size_t channels = in_dim.z;
            size_t size = in_dim.x * in_dim.y * in_dim.z;
            size_t out_size = out_dim.x * out_dim.y * out_dim.z;
            gpu_setZero << <((size + 32) - 1) / 32, 32 >> > (d_error_terms_href[i], size);
            cudaDeviceSynchronize();
            gpu_backPool << <((size + 32) - 1) / 32, 32 >> > (in_dim, d_outputs_href[i + 1], d_error_terms_href[i], out_dim, d_error_terms_href[i + 1], pool_size, padding, stride);
            cudaDeviceSynchronize();
        }
        else if (layers[i + 2] == DENSE) {
            cleanGradient << <(cells_in_layer[i + 1] / 32) + 1, 32 >> > (d_conv_weights_href[i], d_error_terms_href[i + 1], d_error_terms_href[i], d_outputs_href[i + 1], cells_in_layer[i + 2], cells_in_layer[i + 1], A_Fs[i]);
            cudaDeviceSynchronize();
        }
    }

    for (int i = 0;i < cells_in_layer.size() - 1;i++) {
        if (layers[i + 1] == CONV) {
            vec3 in_dim = dimensions[i];
            vec3 out_dim = dimensions[i + 1];
            size_t padding = conv_params[i][0];
            size_t stride = conv_params[i][1];
            vec3 k_dim = vec3(conv_weights[i][0].size(), conv_weights[i][0][0].size(), conv_weights[i][0][0][0].size());
            size_t channels = in_dim.z;
            size_t grad_size = k_dim.x * k_dim.y * k_dim.z * out_dim.z;
            dim3 gridDim(out_dim.z, (grad_size / out_dim.z + 32 - 1) / 32);
            gpu_setZero << <((grad_size + 32) - 1) / 32, 32 >> > (d_conv_gradient_href[i], grad_size);
            cudaDeviceSynchronize();
            gpu_gradVolve<<<gridDim, 32>>> (in_dim, d_outputs_href[i], k_dim, d_conv_gradient_href[i], out_dim, d_error_terms_href[i], padding, stride);
            cudaDeviceSynchronize();
        }
        else if (layers[i + 1] == DENSE) {
            cleanUpdateWeightsbyLayer << <cells_in_layer[i] + 1, cells_in_layer[i + 1] >> > (d_conv_weights_href[i], d_error_terms_href[i], d_outputs_href[i], eta, cells_in_layer[i], cells_in_layer[i + 1], bias);
            cudaDeviceSynchronize();
        }
    }
    for (int i = 0;i < conv_weights.size();i++) {
        if (layers[i + 1] == CONV) {
            vec3 out_dim = dimensions[i + 1];
            vec3 k_dim = vec3(conv_weights[i][0].size(), conv_weights[i][0][0].size(), conv_weights[i][0][0][0].size());
            size_t grad_size = k_dim.x * k_dim.y * k_dim.z * out_dim.z;
            dim3 gridDim(out_dim.z, (grad_size / out_dim.z + 32 - 1) / 32);
            applyGrad << < ((grad_size + 32) - 1) / 32, 32 >> > (d_conv_weights_href[i], d_conv_gradient_href[i], eta, grad_size);
        }
        else if (layers[i + 1] == DENSE) {
            applyGrad << <cells_in_layer[i] + 1, cells_in_layer[i + 1]>> > (d_conv_weights_href[i], d_conv_gradient_href[i], eta, (cells_in_layer[i] + 1) * cells_in_layer[i + 1]);
        }
    }

    float* loss = new float;
    cudaMemcpy(loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
    //this->h_weights = cudaCopy3dBackToVectorHref(&d_weights_href, weight_lengths);
    return *loss;
}

vector<vector<float>> MultiLayerParatron::getCleanerBp(float* x, float* y) {
    int layers = cells_in_layer.size();
    cleanerbp(x, y);
    //return cudaCopy2dBackToVectorHref(d_error_terms_href, vector<int>({ 512,512,10 }));
    //return cudaCopy2dBackToVectorHref(&d_error_terms_href[layers - 1], vector<int>({10}));
    this->h_weights = cudaCopy3dBackToVectorHref(&d_weights_href, weight_lengths);
    //return cudaCopy2dBackToVectorHref(d_weights_href, vector<int>({ 785 * 512,513 * 512,513 * 10 }));
    return { {} };
}

float MultiLayerParatron::aveBatchP(float* batchX, float* batchY) {

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
        batchGradient << <dim3(batchSize, (cells_in_layer[i + 1] / 32) + 1), 32 >> > (d_weights_href[i], d_batch_errors_href[i + 1], d_batch_errors_href[i], d_batch_outs_href[i + 1], cells_in_layer[i + 2], cells_in_layer[i + 1], cells_in_layer[i + 1] * batchSize, A_Fs[i]);
        gpuErrorchk(cudaDeviceSynchronize());
    }

    for (int i = 0;i < cells_in_layer.size() - 1;i++) {
        batchMakeGradient << < dim3(batchSize, cells_in_layer[i] + 1), cells_in_layer[i + 1] >> > (d_batch_grad_href[i], d_batch_errors_href[i], d_batch_outs_href[i], cells_in_layer[i], cells_in_layer[i + 1], bias, batchSize);
        gpuErrorchk(cudaDeviceSynchronize());
    }

    for (int i = 0;i < h_weights.size();i++) {
        averageGrad << <cells_in_layer[i] + 1, cells_in_layer[i + 1], 0, streams[i] >> > (d_batch_grad_href[i], d_gradient_href[i], batchSize, momentum, d_moments_href[i], (cells_in_layer[i] + 1), cells_in_layer[i + 1]);
    }
    gpuErrorchk(cudaDeviceSynchronize());

    for (int i = 0;i < h_weights.size();i++) {
        applyGrad << <cells_in_layer[i] + 1, cells_in_layer[i + 1], 0, streams[i] >> > (d_weights_href[i], d_gradient_href[i], eta, (cells_in_layer[i] + 1) * cells_in_layer[i + 1]);
    }
    gpuErrorchk(cudaDeviceSynchronize());

    float* loss = new float;
    cudaMemcpy(loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
    //this->h_weights = cudaCopy3dBackToVectorHref(&d_weights_href, weight_lengths);

    return *loss;
}

vector<vector<float>> MultiLayerParatron::getAveP(float* batchX, float* batchY) {
    int layers = cells_in_layer.size();
    aveBatchP(batchX, batchY);
    //return cudaCopy2dBackToVectorHref(d_batch_errors_href, vector<int>({ 512,512,10 }));
    //return cudaCopyBatchBackToVectorHref(&d_batch_outs_href[layers - 1], cells_in_layer[layers - 1], batchSize);
    //printf("CHECK WEIGHTS: \n\n");
    //compare3D(this->h_weights, cudaCopy3dBackToVectorHref(&d_weights_href, weight_lengths));
    float* test = new float;
    //cudaMemcpy(test, d_weights_href[0], sizeof(float), cudaMemcpyDeviceToHost);
    //printf("WEIGHT TEST: %f\n", test);
    return cudaCopy2dBackToVectorHref(&d_weights_href, vector<int>({ 785 * 512,513 * 512,513 * 10 }));
}

void MultiLayerParatron::toCPU() {
    this->h_weights = cudaCopy3dBackToVectorHref(&d_weights_href, weight_lengths);
    this->error_terms = cudaCopy2dBackToVectorHref(&d_error_terms_href, vector<int>(cells_in_layer.begin() + 1, cells_in_layer.end()));
    this->outputs = cudaCopy2dBackToVectorHref(&d_outputs_href, cells_in_layer);
}