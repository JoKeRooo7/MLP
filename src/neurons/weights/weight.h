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


================================================================== */


namespace mlp {

namespace graph {

// the Weight class with the error back propagation algorithm
template <typename T>
class Weight {
    public:
        Weight(mlp::TrainingParameters &learning_parametrs);
        Weight& operator=(mlp::TrainingParameters &learning_parametrs);

        template <typename X,typename Y>
        void UpdateWeight(X &error_value, Y &output);
        
        const T *GetWeight();
        const T *Value();
    private:
        T value_;
        T delta_prev_value_{0};
        mlp::TrainingParameters *learning_parametrs_ = nullptr;

        template <typename C>
        void CheckNumericType();
        void InitWeitght();
};  // class Weight

}  // namespace graph

}  // namespace mlp


#include "weight.tpp"


#endif  // MLP_NEURONS_WEIGHTS_H_
