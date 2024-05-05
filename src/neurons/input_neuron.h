#ifndef MLP_NEURONS_INPUT_NEURON_H_
#define MLP_NEURONS_INPUT_NEURON_H_
// вектор как верхушка дерева, которая содержит эти нейроны


#include <vector>
#include <cstddef>


#include "neuron.h"


namespace mlp {

namespace graph {

class InputNeuron : public Neuron {
    public:
        InputNeuron(std::size_t &layer_id, std::size_t &neuron_id);
        void AddParents(std::size_t number_of_children, Neuron first_child);
        void AddChilds(std::size_t number_of_children, Neuron first_child);

    private:
        std::size_t layer_id_;
        std::size_t neuron_id_;
        
        // the weight between this neuron and the child + child
        std::vector<std::pair<float*, Neuron*>> children_;

};  // InputNeuron

}  // graph

}  // mlp

#endif // MLP_NEURONS_INPUT_NEURON_H_
