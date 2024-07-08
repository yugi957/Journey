#include "../Headers/MLP.h"

//MLP class methods:

MultiLayerPerceptron::MultiLayerPerceptron() {

}

void MultiLayerPerceptron::initializeWeights() {
    for (int i = 0; i < cells_in_layer.size() - 1; ++i) {
        float he_initialization = sqrt(2.0 / cells_in_layer[i]);
        for (int j = 0; j < cells_in_layer[i + 1]; ++j) {
            for (int k = 0; k < cells_in_layer[i]; ++k) {
                h_weights[i][j][k] = he_initialization * (rand() / double(RAND_MAX));
            }
            h_weights[i][j][cells_in_layer[i]] = he_initialization * (rand() / double(RAND_MAX)); // bias
        }
    }
}


MultiLayerPerceptron::MultiLayerPerceptron(vector<int> CIL, loss_function func, float bias, float eta, float momentum, int batchSize) {
    this->cells_in_layer = CIL;
    this->L_F = func;
    this->bias = bias;
    this->eta = eta;
    this->batchSize = batchSize;
    this->momentum = momentum;

    for (int i = 0;i < cells_in_layer.size();i++) {
        outputs.push_back(vector<float>(cells_in_layer[i], 0.0));
    }
}

void MultiLayerPerceptron::addLayer(int CIL, activation_function func, vec3 dims) {
    if (dims.x == -1) dims.x = CIL;
    if (cells_in_layer.size() > 0) { //means input layer is done
        int LL = cells_in_layer.size() - 1; //Last Layer index before new layer
        cells_in_layer.push_back(CIL);
        outputs.push_back(vector<float>(CIL, 0.0));
        error_terms.push_back(vector<float>(CIL, 0.0));
        cout << "bytes supposed to be using: " << long int(32 + 8 * CIL * (cells_in_layer[LL] + 1)) << endl;
        h_weights.push_back(vector<vector<float>>(CIL, vector<float>(cells_in_layer[LL] + 1, 0.0)));
        conv_weights.push_back(vector<vector<vector<vector<float>>>>(1, vector<vector<vector<float>>>(1, vector<vector<float>>(CIL, vector<float>(cells_in_layer[LL] + 1, 0.0)))));
        A_Fs.push_back(func);
        dimensions.push_back(dims);
        layers.push_back(DENSE);
        conv_params.push_back({ });
    }
    else {
        cells_in_layer.push_back(CIL);
        outputs.push_back(vector<float>(CIL, 0.0));
        layers.push_back(INPUT);
        dimensions.push_back(dims);
    }
}

void MultiLayerPerceptron::addConv(size_t kernel_size, size_t padding, size_t stride, size_t out_depth, activation_function func) {
    if (cells_in_layer.size() > 0) { //means input layer is done
        int LL = cells_in_layer.size() - 1; //Last Layer index before new layer
        vec3 in = dimensions[LL];
        h_weights.push_back(vector<vector<float>>());
        conv_weights.push_back(vector<vector<vector<vector<float>>>>(out_depth, vector<vector<vector<float>>>(kernel_size, vector<vector<float>>(kernel_size, vector<float>(in.z, 0.0)))));
        vec3 out = getDims(in, kernel_size, padding, stride, out_depth);
        outputs.push_back(vector<float>(out.x*out.y*out.z, 0.0));
        error_terms.push_back(vector<float>(out.x * out.y * out.z, 0.0));
        cells_in_layer.push_back(out.x * out.y * out.z);
        A_Fs.push_back(func);
        dimensions.push_back(out);
        layers.push_back(CONV);
        conv_params.push_back({ padding,stride });
    }
    else printf("INITIALIZE NN WITH INPUT LAYER BEFORE ADDING MORE LAYERS");
}

void MultiLayerPerceptron::addMaxPool(size_t kernel_size, size_t padding = 0, size_t stride = 1) {
    if (cells_in_layer.size() > 0) { //means input layer is done
        int LL = cells_in_layer.size() - 1; //Last Layer index before new layer
        vec3 in = dimensions[LL];
        h_weights.push_back(vector<vector<float>>());
        conv_weights.push_back({{{{float(kernel_size)}}}});
        vec3 out = getDims(in, kernel_size, padding, stride, in.z);
        outputs.push_back(vector<float>(out.x * out.y * out.z, 0.0));
        error_terms.push_back(vector<float>(out.x * out.y * out.z, 0.0));
        cells_in_layer.push_back(out.x * out.y * out.z);
        A_Fs.push_back(PLACEHOLDER);
        dimensions.push_back(out);
        layers.push_back(MAX_POOL);
        conv_params.push_back({ padding,stride });
    }
    else printf("INITIALIZE NN WITH INPUT LAYER BEFORE ADDING MORE LAYERS");
}

void MultiLayerPerceptron::finalize() {
    for (int i = 0;i < batchSize;i++) {
        batch_outputs.push_back(outputs);
        batch_ETs.push_back(error_terms);
        batch_gradients.push_back(h_weights);
    }
    for (int i = 0;i < h_weights.size();i++) {
        gradient.push_back(vector<vector<float>>());
        for (int j = 0;j < h_weights[i].size();j++) {
            gradient[i].push_back(vector<float>(h_weights[i][j].size(), 0));
        }
    }
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
    for (int i = 0;i < h_weights.size();i++) {
        if (layers[i + 1] == DENSE){
            he_init(h_weights[i], h_weights[i][0].size(), h_weights[i].size());
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
    cout << endl;
}

float MultiLayerPerceptron::run(vector<float>& x, vector<float>& w, activation_function A_F, int layer) {
    x.push_back(bias);
    float sum = inner_product(x.begin(), x.end(), w.begin(), (float)0.0);
    x.pop_back();
    return activation(sum, A_F);
}

vector<float> MultiLayerPerceptron::softmax(vector<float> x, vector<vector<float>> w) {
    vector<float> temp(w.size());
    x.push_back(bias);
    float sum = 0.0;
    for (int i = 0;i < w.size();i++) {
        temp[i] = exp(inner_product(x.begin(), x.end(), w[i].begin(), (float)0.0));
        sum += temp[i];
    }
    if (sum < FLT_MIN)
        sum = FLT_MIN;
    if (sum > FLT_MAX)
        sum = FLT_MAX;
    for (int i = 0;i < w.size();i++) {
        temp[i] /= sum;
        if (isnan(temp[i])) 
            printf("w wha\n");
    }
    return temp;
}

float MultiLayerPerceptron::activation(float x, activation_function A_F) {
    switch (A_F) {
    case SIGMOID:
        return 1 / (1 + exp(-x)); //sigmoid
        break;
    case RELU:
        if (x > 0) return x; //ReLu
        return 0;
        break;
    case LEAKY_RELU:
        if (x > 0) return x; //ReLu
        return .1 * x;
        break;
    case SOFTMAX:
        //softmax
        return exp(x);
        break;
    }
}

vector<float> MultiLayerPerceptron::Wrun(vector<float>& x) {
    outputs[0] = x;
    for (int i = 1;i < cells_in_layer.size();i++) {
        if (A_Fs[i - 1] != SOFTMAX)
            for (int j = 0;j < cells_in_layer[i];j++) {
                outputs[i][j] = run(outputs[i - 1], h_weights[i - 1][j], A_Fs[i - 1], i);
                if (isnan(outputs[i][j]))
                    printf("w wha\n");
            }
        else outputs[i] = softmax(outputs[i - 1], h_weights[i - 1]);
    }
    return outputs.back();
}

void MultiLayerPerceptron::activate_conv(vector<float>& x, activation_function A_F) {
    for (auto& val : x) {
        switch (A_F) {
        case SIGMOID:
            val = 1 / (1 + exp(-val)); //sigmoid
            break;
        case RELU:
            if (val <= 0) val = 0; //ReLu
            break;
        case LEAKY_RELU:
            if (val <= 0) val *= .1; //Leaky ReLu
            break;
        }
    }
}

vector<float> MultiLayerPerceptron::forward_conv(vector<float> x) {
    outputs[0] = x;
    for (int i = 1;i < cells_in_layer.size();i++) {
        if (A_Fs[i - 1] == SOFTMAX) {
            outputs[i] = softmax(outputs[i - 1], conv_weights[i - 1][0][0]);
        }
        else if(layers[i] == CONV) {
            convolve_volume(dimensions[i-1], outputs[i-1], vec3(conv_weights[i-1][0].size(), conv_weights[i - 1][0][0].size(),
                            conv_weights[i - 1][0][0][0].size()), conv_weights[i - 1], dimensions[i], outputs[i],
                            conv_params[i-1][0], conv_params[i-1][1]);
            activate_conv(outputs[i], A_Fs[i - 1]);
        }
        else if (layers[i] == MAX_POOL) {
            max_pool(dimensions[i - 1], outputs[i - 1], dimensions[i], outputs[i], size_t(conv_weights[i - 1][0][0][0][0]), conv_params[i - 1][0], conv_params[i - 1][1]);
        }
        else if (layers[i] == DENSE) {
            for (int j = 0;j < cells_in_layer[i];j++) {
                outputs[i][j] = run(outputs[i - 1], conv_weights[i - 1][0][0][j], A_Fs[i - 1], i);
                if (isnan(outputs[i][j]))
                    printf("w wha\n");
            }
        }
    }
    return outputs.back();
}

vector<vector<float>> MultiLayerPerceptron::batchRun(vector<vector<float>> x) {
    vector<vector<float>> out;
    for (int b = 0;b < batchSize;b++) {
        batch_outputs[b][0] = x[b];
        for (int i = 1;i < cells_in_layer.size();i++) {
            if (A_Fs[i - 1] != SOFTMAX)
                for (int j = 0;j < cells_in_layer[i];j++) {
                    batch_outputs[b][i][j] = run(batch_outputs[b][i - 1], h_weights[i - 1][j], A_Fs[i - 1], i);
                }
            else batch_outputs[b][i] = softmax(batch_outputs[b][i - 1], h_weights[i - 1]);
        }
        out.push_back(batch_outputs[b].back());
    }
    return out;
}

vector<vector<float>> MultiLayerPerceptron::Wout(vector<float> x) {
    outputs[0] = x;
    for (int i = 1;i < cells_in_layer.size();i++) {
        if (A_Fs[i - 1] != SOFTMAX)
            for (int j = 0;j < cells_in_layer[i];j++) {
                outputs[i][j] = run(outputs[i - 1], h_weights[i - 1][j], A_Fs[i - 1], i);
            }
        else outputs[i] = softmax(outputs[i - 1], h_weights[i - 1]);
        printArray(&outputs[i][0], outputs[i].size());
    }
    return outputs;
}

float MultiLayerPerceptron::getLoss(vector<float>& x, vector<float>& y) {
    float loss = 0.0;
    switch (L_F) {
    case(MSE):
        for (int i = 0;i < x.size();i++) {
            loss += pow((x[i] - y[i]), 2);
        }
        loss /= x.size();
        break;
    case(CROSS_ENTROPY):
        for (int i = 0;i < x.size();i++) {
            if (x[i] == 0.0) loss -= y[i] * (float)log(.00001);
            else loss -= y[i] * (float)log(x[i]);
        }
        break;
    }
    return loss;
}

float MultiLayerPerceptron::Wbp(vector<float> x, vector<float> y) {
    //get outputs
    clock_t run_start, run_end;
    run_start = clock();
    vector<float> o = Wrun(x);
    if (isnan(o[0]))
        printf("o wha\n");
    run_end = clock();

    //get loss
    clock_t mse_start, mse_end;
    mse_start = clock();
    float loss = getLoss(o, y);
    mse_end = clock();

    if (isnan(loss))
        printf("l wha\n");

    //output error term = o * (1-o) * (y - o)
    clock_t getError_start, getError_end;
    getError_start = clock();
    int s = o.size();
    for (int i = 0; i < o.size(); i++) {
        switch (L_F) {
        case(MSE):
            error_terms.back()[i] = 2 * (o[i] - y[i]);
            break;
        case(CROSS_ENTROPY):
            error_terms.back()[i] = o[i] - y[i];
            break;
        }
        if (isnan(error_terms.back()[i]))
            printf("w wha\n");
    }
    getError_end = clock();
    if (isnan(error_terms.back()[0]))
        printf("w wha\n");

    //k = j+1
    //i = layer
    //hidden layer terms(ji) = o(j) *(1 - o(j)) * sum(w(j->k)*err(k))
    float delta;
    clock_t propError_start, propError_end;
    propError_start = clock();
    for (int i = error_terms.size() - 2;i >= 0;i--) {
        for (int j = 0;j < cells_in_layer[i + 1];j++) {
            float err_sum = 0;
            for (int k = 0;k < cells_in_layer[i + 2];k++) {
                err_sum += h_weights[i + 1][k][j] * error_terms[i + 1][k];
                if (isnan(err_sum))
                    printf("w wha\n");
            }
            if (A_Fs[i] == SIGMOID) error_terms[i][j] = outputs[i + 1][j] * (1 - outputs[i + 1][j]) * err_sum;
            else if (A_Fs[i] == RELU) error_terms[i][j] = (outputs[i + 1][j] > 0) ? err_sum : 0;
            else if (A_Fs[i] == LEAKY_RELU) error_terms[i][j] = (outputs[i + 1][j] > 0) ? err_sum : (.1 * err_sum);
        }
    }
    propError_end = clock();

    //update weights
    //weight += learning rate * error term *
    clock_t weights_start, weights_end;
    weights_start = clock();
    for (int i = 0;i < cells_in_layer.size() - 1;i++) {
        for (int j = 0;j < cells_in_layer[i + 1];j++) {
            for (int k = 0;k < cells_in_layer[i];k++) {
                delta = eta * error_terms[i][j] * outputs[i][k];
                h_weights[i][j][k] -= delta;
                if (isnan(delta))
                    printf("w wha\n");
            }
            delta = eta * error_terms[i][j] * bias;
            h_weights[i][j][cells_in_layer[i]] -= delta;
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

float MultiLayerPerceptron::Mbp(vector<float> x, vector<float> y) {
    //get outputs
    clock_t run_start, run_end;
    run_start = clock();
    vector<float> o = Wrun(x);
    if (isnan(o[0]))
        printf("o wha\n");
    run_end = clock();

    //get loss
    clock_t mse_start, mse_end;
    mse_start = clock();
    float loss = getLoss(o, y);
    mse_end = clock();

    if (isnan(loss))
        printf("l wha\n");

    //output error term = o * (1-o) * (y - o)
    clock_t getError_start, getError_end;
    getError_start = clock();
    int s = o.size();
    for (int i = 0;i < o.size();i++) {
        //vector<float> yo = error_terms.back();
        error_terms.back()[i] = o[i] * (1 - o[i]) * 2 * (o[i] - y[i]);
        if (isnan(error_terms.back()[i]))
            printf("w wha\n");
    }
    getError_end = clock();

    //k = j+1
    //i = layer
    //hidden layer terms(ji) = o(j) *(1 - o(j)) * sum(w(j->k)*err(k))
    float delta;
    clock_t propError_start, propError_end;
    propError_start = clock();
    for (int i = error_terms.size() - 2;i >= 0;i--) {
        for (int j = 0;j < cells_in_layer[i + 1];j++) {
            float err_sum = 0;
            for (int k = 0;k < cells_in_layer[i + 2];k++) {
                err_sum += h_weights[i + 1][k][j] * error_terms[i + 1][k];
                if (isnan(err_sum))
                    printf("w wha\n");
            }
            if (A_Fs[i] == SIGMOID) error_terms[i][j] = outputs[i + 1][j] * (1 - outputs[i + 1][j]) * err_sum;
            else if (A_Fs[i] == RELU) error_terms[i][j] = (outputs[i + 1][j] > 0) ? err_sum : 0;
            else if (A_Fs[i] == LEAKY_RELU) error_terms[i][j] = (outputs[i + 1][j] > 0) ? err_sum : (.1 * err_sum);
        }
    }
    propError_end = clock();

    //update weights
    //weight += learning rate * error term *
    clock_t weights_start, weights_end;
    weights_start = clock();
    for (int i = 0;i < cells_in_layer.size() - 1;i++) {
        for (int j = 0;j < cells_in_layer[i + 1];j++) {
            for (int k = 0;k < cells_in_layer[i];k++) {
                delta = eta * error_terms[i][j] * outputs[i][k];
                gradient[i][j][k] = momentum * gradient[i][j][k] + delta;
                if (isnan(delta))
                    printf("w wha\n");
            }
            delta = eta * error_terms[i][j] * bias;
            gradient[i][j][cells_in_layer[i]] = momentum * gradient[i][j][cells_in_layer[i]] + delta;
        }
    }

    for (int i = 0;i < cells_in_layer.size() - 1;i++) {
        for (int j = 0;j < cells_in_layer[i + 1];j++) {
            for (int k = 0;k < cells_in_layer[i];k++) {
                h_weights[i][j][k] -= gradient[i][j][k];
                //if (isnan(delta))
                //    printf("w wha\n");
            }
            h_weights[i][j][cells_in_layer[i]] -= gradient[i][j][cells_in_layer[i]];
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

float MultiLayerPerceptron::backward_conv(vector<float> x, vector<float> y) {
    // Get outputs
    clock_t run_start, run_end;
    run_start = clock();
    vector<float> o = forward_conv(x);
    if (isnan(o[0]))
        printf("o wha\n");
    run_end = clock();

    // Get loss
    clock_t mse_start, mse_end;
    mse_start = clock();
    float loss = getLoss(o, y);
    mse_end = clock();

    if (isnan(loss))
        printf("l wha\n");

    // Output error term = o * (1-o) * (y - o)
    clock_t getError_start, getError_end;
    getError_start = clock();
    int s = o.size();
    for (int i = 0; i < o.size(); i++) {
        switch (L_F) {
        case(MSE):
            error_terms.back()[i] = (2.0 / o.size()) * (o[i] - y[i]);
            if (A_Fs.back() == SIGMOID) error_terms.back()[i] *= o[i] * (1 - o[i]);
            else if (A_Fs.back() == RELU) error_terms.back()[i] *= (o[i] > 0) ? 1 : 0;
            else if (A_Fs.back() == LEAKY_RELU) error_terms.back()[i] *= (o[i] > 0) ? 1 : .1;
            break;
        case(CROSS_ENTROPY):
            error_terms.back()[i] = o[i] - y[i];
            break;
        }
        if (isnan(error_terms.back()[i]))
            printf("w wha\n");
    }
    getError_end = clock();

    // Propagate error through layers
    float delta;
    clock_t propError_start, propError_end;
    propError_start = clock();
    for (int i = error_terms.size() - 2; i >= 0; i--) {
        if (layers[i + 2] == CONV) {
            vec3 in_dim = dimensions[i + 1];
            vec3 out_dim = dimensions[i + 2];
            size_t padding = conv_params[i + 1][0];
            size_t stride = conv_params[i + 1][1];
            vec3 k_dim = vec3(conv_weights[i + 1][0].size(), conv_weights[i + 1][0][0].size(), conv_weights[i + 1][0][0][0].size());
            size_t channels = in_dim.z;
            backVolve(in_dim, error_terms[i], k_dim, conv_weights[i + 1], out_dim, error_terms[i + 1], padding, stride);
            for (int j = 0;j < error_terms[i].size();j++) {
                if (A_Fs[i] == SIGMOID) error_terms[i][j] = outputs[i + 1][j] * (1 - outputs[i + 1][j]) * error_terms[i][j];
                else if (A_Fs[i] == RELU) error_terms[i][j] = (outputs[i + 1][j] > 0) ? error_terms[i][j] : 0;
                else if (A_Fs[i] == LEAKY_RELU) error_terms[i][j] = (outputs[i + 1][j] > 0) ? error_terms[i][j] : (.1 * error_terms[i][j]);
            }
        }
        else if (layers[i + 2] == MAX_POOL) {
            vec3 in_dim = dimensions[i + 1];
            vec3 out_dim = dimensions[i + 2];
            size_t pool_size = conv_weights[i+1][0][0][0][0];
            size_t padding = conv_params[i + 1][0];
            size_t stride = conv_params[i + 1][1];
            size_t channels = in_dim.z;
            backPool(in_dim, outputs[i + 1], error_terms[i], out_dim, error_terms[i + 1], pool_size, padding, stride);
        }
        else if (layers[i + 2] == DENSE) {
            for (int j = 0; j < cells_in_layer[i + 1]; j++) {
                float err_sum = 0;
                for (int k = 0; k < cells_in_layer[i + 2]; k++) {
                    err_sum += conv_weights[i + 1][0][0][k][j] * error_terms[i + 1][k];
                    if (isnan(err_sum))
                        printf("w wha\n");
                }
                if (A_Fs[i] == SIGMOID) error_terms[i][j] = outputs[i + 1][j] * (1 - outputs[i + 1][j]) * err_sum;
                else if (A_Fs[i] == RELU) error_terms[i][j] = (outputs[i + 1][j] > 0) ? err_sum : 0;
                else if (A_Fs[i] == LEAKY_RELU) error_terms[i][j] = (outputs[i + 1][j] > 0) ? err_sum : (.1 * err_sum);
            }
        }
    }
    propError_end = clock();

    // Update weights for convolutional and dense layers
    clock_t weights_start, weights_end;
    weights_start = clock();
    for (int i = 0; i < cells_in_layer.size() - 1; i++) {
        if (layers[i + 1] == CONV) {
            vec3 in_dim = dimensions[i];
            vec3 out_dim = dimensions[i + 1];
            size_t padding = conv_params[i][0];
            size_t stride = conv_params[i ][1];
            vec3 k_dim = vec3(conv_weights[i][0].size(), conv_weights[i][0][0].size(), conv_weights[i][0][0][0].size());
            size_t channels = in_dim.z;
            gradVolve(in_dim, outputs[i], k_dim, conv_gradient[i], out_dim, error_terms[i], padding, stride);
            for (int j = 0;j < conv_weights[i].size();j++) {
                for (int k = 0;k < conv_weights[i][j].size();k++) {
                    for (int l = 0;l < conv_weights[i][j][k].size();l++) {
                        for (int m = 0;m < conv_weights[i][j][k][l].size();m++) {
                            conv_weights[i][j][k][l][m] -= eta * conv_gradient[i][j][k][l][m];
                        }
                    }
                }
            }
        }
        else if (layers[i + 1] == DENSE) {
            for (int j = 0; j < cells_in_layer[i + 1]; j++) {
                for (int k = 0; k < cells_in_layer[i]; k++) {
                    delta = error_terms[i][j] * outputs[i][k];
                    conv_gradient[i][0][0][j][k] = momentum * conv_gradient[i][0][0][j][k] + delta;
                    if (isnan(delta))
                        printf("w wha\n");
                }
                delta = error_terms[i][j] * bias;
                conv_gradient[i][0][0][j][cells_in_layer[i]] = momentum * conv_gradient[i][0][0][j][cells_in_layer[i]] + delta;
            }

            for (int j = 0; j < cells_in_layer[i + 1]; j++) {
                for (int k = 0; k < cells_in_layer[i]; k++) {
                    conv_weights[i][0][0][j][k] -= conv_gradient[i][0][0][j][k] * eta;
                }
                conv_weights[i][0][0][j][cells_in_layer[i]] -= conv_gradient[i][0][0][j][cells_in_layer[i]] * eta;
            }
        }
    }
    weights_end = clock();

    return loss;
}

void MultiLayerPerceptron::train(vector<vector<float>> train_set, vector<vector<float>> label_set, int epochs, int progressCheck) {
    float loss = 0.0;
    clock_t gpu_start, gpu_end;
    gpu_start = clock();
    for (int j = 0;j < epochs;j++) {
        for (int i = 0;i < label_set.size();i++) {
            //temp[train_lbls[i][0]] = 1;
            //compare3D(mlp->h_weights, momlp->h_weights);
            loss += this->Mbp(train_set[i], label_set[i]);
            //temp[train_lbls[i][0]] = 0;
            //cout << i << " : " << MSE << endl;
            if (i % progressCheck == 0) {
                gpu_end = clock();
                cout << "ground truth example: " << i << " error: " << loss / progressCheck << endl;
                printExecution("Time taken", gpu_start, gpu_end);
                gpu_start = clock();
                loss = 0.0;
            }
        }
    }
}