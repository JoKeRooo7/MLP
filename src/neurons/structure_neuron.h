#ifndef MLP_NEURONS_STRUCTURE_NEURON_H_
#define MLP_NEURONS_STRUCTURE_NEURON_H_


#include <vector>
#include <cstddeff>  // std::size_t


#include "../../support_functions/saving_data.h"
#include "../../support_functions/update_weight.h"
#include "../../support_functions/activation_function.h"

/* -----------==============================-------------
This is a simple neuron model used to build 
a perceptron.

In the Neuron class:

private fields:
δ (error_) - the value of the error on the neuron
o (output_value_) - the output value of the neuron
w (weight) - weight between other neurons
--------------==============================---------- */


namespace mlp {

namespace graph {

// C - module For sending signals and copying data outside the neuron
template <typename C>
class Neuron {
    public:
        // Neuron() = default;
        explicit Neuron(std::size_t &neurol_id, std::size_t &layer_id, mlp::graph::WeightHandler *ready_module);
        ~Neuron() = default;


        void CalculateError();

    // protected:

    private:
        float error_;  // δ
        float output_value_;  // o


        std::size_t layer_id_;
        std::size_t neuron_id_;
        // weights at the previous update
        std::vector<std::pair<float> prev_weight_children_;
        // the error of the scales on the previous update of the scales
        std::vector<std::pair<float> error_prev_weight_;
        // the weight between this neuron and the parent + parent
        std::vector<std::pair<float*, Neuron<C>*>> parent_;
        // the weight between this neuron and the child + child
        std::vector<std::pair<float, Neuron<C>*>> children_;

        mlp::graph::WeightHandler *module_weith_handler;
};


}  // graph

}  // mlp


#endif  // MLP_NEURONS_STRUCTURE_NEURON_H_
