#ifndef MLP_SUPPORT_FUNCTIONS_ACTIVATION_FUNCTION_H_
#define MLP_SUPPORT_FUNCTIONS_ACTIVATION_FUNCTION_H_


#include <cmath>


namespace mlp {

float sigmoid_function(float x) {
    return 1 / (1 + std::exp(-x));
}

}  // mlp

#endif  // MLP_SUPPORT_FUNCTIONS_ACTIVATION_FUNCTION_H_
