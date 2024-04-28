#ifndef MLP_NEURONS_STRUCTURE_NEURON_H_
#define MLP_NEURONS_STRUCTURE_NEURON_H_


#include <vector>
#include <cstddeff>  // std::size_t


namespace mlp {

namespace graph {

namespace abstract {

// imput info in C module
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
        std::vector<std::pair<float, Neuron<C>>> children_;  // w
        std::vector<std::pair<float, Neuron<C>>> prev_w_children_;
        
};

}  // abstract

}  // graph

}  // mlp


#endif  // MLP_NEURONS_STRUCTURE_NEURON_H_
