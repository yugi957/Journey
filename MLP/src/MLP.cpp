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


MultiLayerPerceptron::MultiLayerPerceptron(vector<int> CIL, loss_function func, float bias, float eta, int batchSize, float momentum) {
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

void MultiLayerPerceptron::addLayer(int CIL, activation_function func) {
    if (cells_in_layer.size() > 0) { //means input layer is done
        int LL = cells_in_layer.size() - 1; //Last Layer index before new layer
        cells_in_layer.push_back(CIL);
        outputs.push_back(vector<float>(CIL, 0.0));
        error_terms.push_back(vector<float>(CIL, 0.0));
        h_weights.push_back(vector<vector<float>>(CIL, vector<float>(cells_in_layer[LL] + 1, 0.0)));
        A_Fs.push_back(func);
    }
    else printf("INITIALIZE NN WITH INPUT LAYER BEFORE ADDING MORE LAYERS");
}

void MultiLayerPerceptron::addConv(dim3 dims, int padding, activation_function func) {
    if (cells_in_layer.size() > 0) { //means input layer is done
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
    for (int i = 0;i < h_weights.size();i++) {
        xavier_init(h_weights[i], h_weights[i][0].size(), h_weights[i].size());
    }
}

float MultiLayerPerceptron::run(vector<float> x, vector<float> w, activation_function A_F, int layer) {
    x.push_back(bias);
    float sum = inner_product(x.begin(), x.end(), w.begin(), (float)0.0);
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
    if (sum < FLT_MAX)
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
        return .1 * x;
        break;
    case SOFTMAX:
        //softmax
        return exp(x);
        break;
    }
}

vector<float> MultiLayerPerceptron::Wrun(vector<float> x) {
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

float MultiLayerPerceptron::getLoss(vector<float> x, vector<float> y) {
    float loss = 0.0;
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
            if (x[i] == 0.0) loss -= y[i] * (float)log(.00001);
            else loss -= y[i] * (float)log(x[i]);
        }
        return loss;
        break;
    }
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
    for (int i = 0;i < o.size();i++) {
        //vector<float> yo = error_terms.back();
        //error_terms.back()[i] = o[i] * (1 - o[i]) * 2 * (o[i] - y[i]);
        error_terms.back()[i] = o[i] - y[i];
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
            else if (A_Fs[i] == RELU) error_terms[i][j] = (outputs[i + 1][j] > 0) ? err_sum : (.1 * err_sum);
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
            else if (A_Fs[i] == RELU) error_terms[i][j] = (outputs[i + 1][j] > 0) ? err_sum : (.1 * err_sum);
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