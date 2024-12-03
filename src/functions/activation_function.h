#ifndef MLP_PERCEPTRON_FUNCTIONS_ACTIVATION_FUNCtION_H_
#define MLP_PERCEPTRON_FUNCTIONS_ACTIVATION_FUNCtION_H_


#include <cmath>


namespace mlp {


    float sigmoid_function(float &x) {
        return 1 / (1 + std::exp(-x));
    }

å
}


#endif  // MLP_PERCEPTRON_FUNCTIONS_ACTIVATION_FUNCtION_H_
