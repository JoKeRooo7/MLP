#include "edge.h"
#include "../neurons/template_neuron.h"


namespace mlp {

    template <typename W>
    Edge<W>::Edge(Neuron *left_neuron) {
        left_neuron_ = left_neuron;
    }

    template <typename W>
    Edge<W>::Edge(Neuron *left_neuron, Neuron *right_neuron) {
        left_neuron_ = left_neuron;
        right_neuron_ = right_neuron;
    }

    template <typename W>
    void Edge<W>::AddLeftNeuron(Neuron *left_neuron) {
        left_neuron_ = left_neuron;
        ResetWeight();
    }

    template <typename W>
    void Edge<W>::AddRightNeuron(Neuron *right_neuron) {
        right_neuron_ = right_neuron;
        ResetWeight();
    }

    template <typename W>
    Edge<W>::~Edge() {
        *left_neuron_ = nullptr;
        *right_neuron_ = nullptr;
    }

    template <typename W>
    void Edge<W>::ResetWeight() {
        this_weight_.Reset();
    }


}  // mlp
