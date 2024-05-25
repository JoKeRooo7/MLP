#include <vector>
#include <cstddef>


#include "neuron.cc"


namespace mlp {

namespace graph {

InputNeuron::InputNeuron(std::size_t &layer_id, std::size_t &neuron_id) {
    layer_id_ = layer_id;
    neuron_id_ = neuron_id;
}

}  // graph

}  // mlp
