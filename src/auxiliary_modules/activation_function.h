#ifndef MLP_AUXILIARY_MODULES_ACTIVATION_FUNCTION_H_
#define MLP_AUXILIARY_MODULES_ACTIVATION_FUNCTION_H_


#include <cmath>


namespace mlp {

float sigmoid_function(float x) {
    return 1 / (1 + std::exp(-x));
}

}  // mlp

#endif  // MLP_AUXILIARY_MODULES_ACTIVATION_FUNCTION_H_
