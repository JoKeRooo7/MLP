#include "edge.h"
#include "../neurons/template_neuron.h"


namespace mlp {


    Edge::Edge(Neuron *left_neuron) {
        left_neuron_ = left_neuron;
    }


    Edge::Edge(Neuron *left_neuron, Neuron *right_neuron) {
        left_neuron_ = left_neuron;
        right_neuron_ = right_neuron;
    }


    void Edge<::AddLeftNeuron(Neuron *left_neuron) {
        left_neuron_ = left_neuron;
        ResetWeight();
    }


    void Edge::AddRightNeuron(Neuron *right_neuron) {
        right_neuron_ = right_neuron;
        ResetWeight();
    }


    void Edge::ComputeAllOutput() {
        right_neuron_ -> ComputeAllOutput();
    }


    const float& Edge::GetWeight() {
        return this_weight_.value;
    }


    const float& GetLeftOutput() {
        return left_neuron_ -> GetOutput();
    }


    const float& GetRightOutput() {
        return right_neuron_ -> GetOutput();
    }


    const float&  GetLeftError() {
        return right_neuron_ -> GetError();
    }

    const float&  GetRightError() {
        return right_neuron_ -> GetError();
    }

    // Edge::~Edge() {
    //     *left_neuron_ = nullptr;
    //     *right_neuron_ = nullptr;
    // }

    // template <typename W>
    // void Edge<W>::ResetWeight() {
    //     this_weight_.Reset();
    // }


}  // mlp
