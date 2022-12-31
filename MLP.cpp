#include "MLP.h"

//perceptron class methods:
double frand() {
    return (2.0 * (double)rand() / RAND_MAX) - 1.0;
}

//return new perceptron with number of inputs + 1 bias
Perceptron::Perceptron(int inputs, double bias) {
    this->bias = bias;
    weights.resize(inputs + 1);
    generate(weights.begin(), weights.end(), frand);

}

//Run the perceptron with x as input
double Perceptron::run(vector<double> x) {
    x.push_back(bias);
    double sum = inner_product(x.begin(), x.end(), weights.begin(), (double)0.0);
    return activation(sum);

}

double Perceptron::activation(double x) {
    return 1 / (1 + exp(-x)); //sigmoid

    //ReLu
    //if (x > 0) return x;
    //else return 0;

}

void Perceptron::set_weights(vector<double> w_init) {
    weights = w_init;
}

//MLP class methods:
MultiLayerPerceptron::MultiLayerPerceptron(vector<int> CIL, double bias, double eta) {
    this->cells_in_layer = CIL;
    this->bias = bias;
    this->eta = eta;

    for (int i = 0;i < cells_in_layer.size();i++) {
        outputs.push_back(vector<double>(cells_in_layer[i], 0.0));
        error_terms.push_back(vector<double>(cells_in_layer[i], 0.0));
        network.push_back(vector<Perceptron>());
        //clock_t gpu_start, gpu_end;
        if (i > 0) { //input layer has no neurons
            for (int j = 0;j < cells_in_layer[i];j++) {
                //gpu_start = clock();
                network[i].push_back(Perceptron(cells_in_layer[i - 1], bias));
                //gpu_end = clock();
                //printf("%d : %d ", cells_in_layer[i], j);
                //printExecution("Loop", gpu_start, gpu_end);
            }
        }
    }
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

vector<double> MultiLayerPerceptron::run(vector<double> x) {
    outputs[0] = x;
    for (int i = 1;i < network.size();i++) {
        for (int j = 0;j < cells_in_layer[i];j++) {
            outputs[i][j] = network[i][j].run(outputs[i - 1]);
        }
    }
    return outputs.back();
}

double MultiLayerPerceptron::bp(vector<double> x, vector<double> y) {
    //get outputs
    vector<double> o = run(x);

    //get mse
    double mse = 0;
    for (int i = 0;i < o.size();i++) {
        mse += pow((y[i] - o[i]), 2);
    }
    mse /= o.size();

    //output error term = o * (1-o) * (y - o)
    int s = o.size();
    for (int i = 0;i < s;i++) {
        vector<double> yo = error_terms.back();
        error_terms.back()[i] = o[i] * (1 - o[i]) * (y[i] - o[i]);
    }

    //k = j+1
    //i = layer
    //hidden layer terms(ji) = o(j) *(1 - o(j)) * sum(w(j->k)*err(k))
    for (int i = error_terms.size() - 2;i >= 0;i--) {
        for (int j = 0;j < error_terms[i].size();j++) {
            double err_sum = 0;
            for (int k = 0;k < cells_in_layer[i + 1];k++) {
                err_sum += network[i + 1][k].weights[j] * error_terms[i + 1][k];
            }
            error_terms[i][j] = outputs[i][j] * (1 - outputs[i][j]) * err_sum;
        }
    }

    //update weights
    //weight += learning rate * error term *
    for (int i = 1;i < cells_in_layer.size();i++) {
        for (int j = 0;j < cells_in_layer[i];j++) {
            for (int k = 0;k < cells_in_layer[i - 1] + 1;k++) {
                double delta;
                if (k == cells_in_layer[i - 1])
                    delta = eta * error_terms[i][j] * bias;
                else
                    delta = eta * error_terms[i][j] * outputs[i - 1][k];
                network[i][j].weights[k] += delta;
            }
        }
    }
    return mse;


}
