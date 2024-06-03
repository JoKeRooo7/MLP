#ifndef MLP_NEURONS_WEIGHTS_H_
#define MLP_NEURONS_WEIGHTS_H_


#include <typeinfo>
#include <stdexcept>


#include "../../auxiliary_modules/training_parameters.h"


/* ==================================================================
This module is the manipulation of weights and the storage 
of each neuron's day weight, as well as its previous values.

In the Weight class:

private fields:
w (value_) - weight between other neurons
w_(x-1) (prev_value_) - previous weight value
dw (delta_weight) - weight between other neurons
================================================================== */


namespace mlp {

namespace graph {

// the Weight class with the error back propagation algorithm
template <typename T>
class Weight {
    public:
        Weight(mlp::TrainingParameters &learning_parametrs);
        Weight& operator=(mlp::TrainingParameters &learning_parametrs);

        void UpdateWeight(float &error_value, float &output);

        const float *GetWeight();
        const float *Value();
    private:
        using Numeric = T;
        Numeric value_;
        Numeric delta_prev_value_{0};
        mlp::TrainingParameters *learning_parametrs_ = nullptr;

        void InitWeitght();

};  // Weight

}  // graph

}  // mlp


#include "weight.tpp"


#endif  // MLP_NEURONS_WEIGHTS_H_
