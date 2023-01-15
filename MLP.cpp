#include "MLP.h"

//perceptron class methods:
Perceptron::Perceptron() {

}

//return new perceptron with number of inputs + 1 bias
Perceptron::Perceptron(int inputs, activation_function func, double bias) {
    this->bias = bias;
    this->A_F = func;
    weights.resize(inputs + 1);
    generate(weights.begin(), weights.end(), frand);

}

//Run the perceptron with x as input
double Perceptron::run(vector<double> x) {
    x.push_back(bias);
    vector<double> temp;
    double sum = inner_product(x.begin(), x.end(), weights.begin(), (double)0.0);
    return activation(sum);
}

double Perceptron::activation(double x) {
    switch (A_F) {
        case SIGMOID:
            return 1 / (1 + exp(-x)); //sigmoid
            break;
        case RELU:
            if (x > 0) return 1; //ReLu
            return 0;
            break;
        case SOFTMAX:
            //softmax
            return exp(x);
            break;
    }
}

void Perceptron::set_weights(vector<double> w_init) {
    weights = w_init;
}

//MLP class methods:

MultiLayerPerceptron::MultiLayerPerceptron() {

}

MultiLayerPerceptron::MultiLayerPerceptron(vector<int> CIL, loss_function func, double bias, double eta) {
    this->cells_in_layer = CIL;
    this->L_F = func;
    this->bias = bias;
    this->eta = eta;

    for (int i = 0;i < cells_in_layer.size();i++) {
        outputs.push_back(vector<double>(cells_in_layer[i], 0.0));
        //error_terms.push_back(vector<double>(cells_in_layer[i], 0.0));
        network.push_back(vector<Perceptron>());
        //clock_t gpu_start, gpu_end;
        if (i > 0) { //input layer has no neurons
            for (int j = 0;j < cells_in_layer[i];j++) {
                network[i].push_back(Perceptron(cells_in_layer[i - 1], SIGMOID, bias));
            }
        }
    }
}

void MultiLayerPerceptron::addLayer(int CIL, activation_function func) {
    if (cells_in_layer.size() > 0) { //means input layer is done
        int LL = cells_in_layer.size() - 1; //Last Layer index before new layer
        cells_in_layer.push_back(CIL);
        outputs.push_back(vector<double>(CIL, 0.0));
        error_terms.push_back(vector<double>(CIL, 0.0));
        h_weights.push_back(vector<vector<double>>(CIL, vector<double>(cells_in_layer[LL]+1, 0.0)));
        A_Fs.push_back(func);
        //for (int i = 0;i < CIL;i++) {
        //    generate(h_weights[LL][i].begin(), h_weights[LL][i].end(), frand);
        //}
        network.push_back(vector<Perceptron>());
        for (int i = 0;i < CIL;i++) {
            network[LL + 1].push_back(Perceptron(cells_in_layer[LL], func, bias));
        }
        for (int j = 0;j < CIL;j++) {
            h_weights[LL][j] = network[LL+1][j].weights;
        }
    }
    else printf("INITIALIZE NN WITH INPUT LAYER BEFORE ADDING MORE LAYERS");
}

void MultiLayerPerceptron::set_weights(vector<vector<vector<double>>> w_init) {
    for (int i = 0;i < w_init.size();i++) {
        for (int j = 0;j < w_init[i].size();j++) {
            network[i + 1][j].set_weights(w_init[i][j]);
        }
    }
}

void MultiLayerPerceptron::print_weights() {
    cout << endl;
    for (int i = 0;i < network.size();i++) {
        for (int j = 0;j < network[i].size();j++) {
            cout << "Layer: " << i + 1 << " Neuron " << j << ": ";
            for (auto& it : network[i][j].weights) {
                cout << it << "   ";
            }
            cout << endl;
        }
    }
    cout << endl;
}

double MultiLayerPerceptron::run(vector<double> x, vector<double> w, activation_function A_F, int layer) {
    x.push_back(bias);
    double sum = inner_product(x.begin(), x.end(), w.begin(), (double)0.0);
    return activation(sum, A_F);
}

vector<double> MultiLayerPerceptron::softmax(vector<double> x, vector<vector<double>> w) {
    vector<double> temp(w.size());
    x.push_back(bias);
    double sum = 0.0;
    for (int i = 0;i < w.size();i++) {
        temp[i] = exp(inner_product(x.begin(), x.end(), w[i].begin(), (double)0.0));
        sum += temp[i];
    }
    if (sum < DBL_MIN)
        sum = DBL_MIN;
    for (int i = 0;i < w.size();i++) {
        temp[i] /= sum;
    }
    return temp;
}

double MultiLayerPerceptron::activation(double x, activation_function A_F) {
    switch (A_F) {
    case SIGMOID:
        return 1 / (1 + exp(-x)); //sigmoid
        break;
    case RELU:
        if (x > 0) return 1; //ReLu
        return 0;
        break;
    case SOFTMAX:
        //softmax
        return exp(x);
        break;
    }
}

vector<double> MultiLayerPerceptron::run(vector<double> x) {
    outputs[0] = x;
    for (int i = 1;i < network.size();i++) {
        for (int j = 0;j < cells_in_layer[i];j++) {
            outputs[i][j] = network[i][j].run(outputs[i - 1]);
        }
    }
    if (network[network.size()-1][0].A_F == SOFTMAX) {
        vector<double> last = outputs.back();
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

vector<double> MultiLayerPerceptron::Wrun(vector<double> x) {
    outputs[0] = x;
    for (int i = 1;i < network.size();i++) {
        if (A_Fs[i - 1] != SOFTMAX)
            for (int j = 0;j < cells_in_layer[i];j++) {
                outputs[i][j] = run(outputs[i - 1], h_weights[i - 1][j], A_Fs[i - 1], i);
            }
        else outputs[i] = softmax(outputs[i - 1], h_weights[i - 1]);
    }
    return outputs.back();
}

vector<vector<double>> MultiLayerPerceptron::Wout(vector<double> x) {
    outputs[0] = x;
    for (int i = 1;i < network.size();i++) {
        if (A_Fs[i - 1] != SOFTMAX)
            for (int j = 0;j < cells_in_layer[i];j++) {
                outputs[i][j] = run(outputs[i - 1], h_weights[i - 1][j], A_Fs[i - 1], i);
            }
        else outputs[i] = softmax(outputs[i - 1], h_weights[i - 1]);
        printArray(&outputs[i][0], outputs[i].size());
    }
    return outputs;
}

double MultiLayerPerceptron::getLoss(vector<double> x, vector<double> y) {
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

double MultiLayerPerceptron::bp(vector<double> x, vector<double> y) {
    //get outputs
    clock_t run_start, run_end;
    run_start = clock();
    vector<double> o = run(x);
    if (isnan(o[0]))
        printf("o wha\n");
    run_end = clock();

    //get loss
    clock_t mse_start, mse_end;
    mse_start = clock();
    double loss = getLoss(o, y);
    mse_end = clock();

    if (isnan(loss))
        printf("l wha\n");

    //output error term = o * (1-o) * (y - o)
    clock_t getError_start, getError_end;
    getError_start = clock();
    int s = o.size();
    for (int i = 0;i < s;i++) {
        //vector<double> yo = error_terms.back();
        error_terms.back()[i] = o[i] * (1 - o[i]) * (y[i] - o[i]);
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
                err_sum += network[i + 1][k].weights[j] * error_terms[i + 1][k];
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
                network[i][j].weights[k] += delta;
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

double MultiLayerPerceptron::Wbp(vector<double> x, vector<double> y) {
    //get outputs
    clock_t run_start, run_end;
    run_start = clock();
    vector<double> o = Wrun(x);
    if (isnan(o[0]))
        printf("o wha\n");
    run_end = clock();

    //get loss
    clock_t mse_start, mse_end;
    mse_start = clock();
    double loss = getLoss(o, y);
    mse_end = clock();

    if (isnan(loss))
        printf("l wha\n");

    //output error term = o * (1-o) * (y - o)
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
    double delta;
    clock_t propError_start, propError_end;
    propError_start = clock();
    for (int i = error_terms.size() - 2;i >= 0;i--) {
        for (int j = 0;j < cells_in_layer[i + 1];j++) {
            double err_sum = 0;
            for (int k = 0;k < cells_in_layer[i + 2];k++) {
                err_sum += h_weights[i+1][k][j] * error_terms[i + 1][k];
                if (isnan(err_sum))
                    printf("w wha\n");
            }
            error_terms[i][j] = outputs[i+1][j] * (1 - outputs[i+1][j]) * err_sum;
        }
    }
    propError_end = clock();

    //update weights
    //weight += learning rate * error term *
    clock_t weights_start, weights_end;
    weights_start = clock();
    for (int i = 0;i < cells_in_layer.size()-1;i++) {
        for (int j = 0;j < cells_in_layer[i+1];j++) {
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
