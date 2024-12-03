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
    void Edge<W>::ComputeAllOutput() {
        right_neuron_ -> ComputeAllOutput();
    }

    template <typename W>
    const float& Edge<W>::GetWeight() {
        return this_weight_.value;
    }

    template <typename W>
    const float& GetLeftOutput() {
        return left_neuron_ -> GetOutput();
    }

    template <typename W>
    const float& GetRightOutput() {
        return right_neuron_ -> GetOutput();
    }

    template <typename W>
    Edge<W>::~Edge() {
        *left_neuron_ = nullptr;
        *right_neuron_ = nullptr;
    }

    // template <typename W>
    // void Edge<W>::ResetWeight() {
    //     this_weight_.Reset();
    // }


}  // mlp
