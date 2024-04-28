#ifndef MLP_NEURONS_STRUCTURE_NEURON_H_
#define MLP_NEURONS_STRUCTURE_NEURON_H_


#include <vector>
#include <cstddeff>  // std::size_t


#include "../../support_functions/saving_data.h"
#include "../../support_functions/activation_function.h"

/* -----------==============================-------------
This is a simple neuron model used to build 
a perceptron.

δ (error_) - the value of the error on the neuron
o (output_value_) - the output value of the neuron
η (learning_rate_) - learning rate
w (weight) - weight between other neurons
α (inertia_coefficient_) -  is the inertia 
coefficient to smooth out sharp overshoots 
as you move across the surface of the objective function.
--------------==============================---------- */


namespace mlp {

namespace graph {

namespace abstract {

// C - module For sending signals and copying data outside the neuron
template <typename C>
class Neuron {
    public:

    protected:

    private:
        float error_;  // δ
        float output_value_;  // o
        float learning_rate_;  // η
        
        std::size_t layer_id_;
        std::size_t neuron_id_;
        std::vector<std::pair<float> prev_w_children_;  // w
        std::vector<std::pair<float> error_prev_weight_;  // Δw
        std::vector<std::pair<float*, Neuron<C>*>> parent_;  // w
        std::vector<std::pair<float*, Neuron<C>*>> children_;  // w
};

}  // abstract

}  // graph

}  // mlp


#endif  // MLP_NEURONS_STRUCTURE_NEURON_H_
