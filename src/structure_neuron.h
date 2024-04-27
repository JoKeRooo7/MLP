#ifndef MLP_GRAPH_NEURON_
#define MLP_GRAPH_NEURON_

#include <vector>
#include <cstddeff>  // std::size_t


namespace mlp {

namespace abstract {

// imput info in C module
template <typename C>
class Neuron {
    public:

    protected:

    private:
        float error;
        std::size_t id_;
        std::vector<std::pair<float, Neuron<C>>> children_;
};

} //  abstract

} //  MLP

#endif  // MLP_GRAPH_NEURON_