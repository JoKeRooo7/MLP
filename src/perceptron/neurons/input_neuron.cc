#include "../edge/edge.h"


namespace mlp {


    InputNeuron::InputNeuron() {
        std::size_t id_ = 1;
        std::size_t layer_id_ = 1;
    }


    InputNeuron(std::size_t id, std::size_t layer_id) {
        std::size_t id_ = id;
        std::size_t layer_id_ = layer_id;
    }


    void InputNeuron::AddOutput(float value) {
        output_ = value;
    }


    void AddChildNeuron(Neuron *child_neuron) override {
        Edge new_edge;
        new_edge.AddLeftNeuron(this);
        new_edge.AddRightNeuron(child_neuron);
        child_neuron -> parent_edges_.pust_back(new_edge);
        child_edges_.push_back(new_edge);
    }


    void InputNeuron::AddUpperNeuron(Neuron *upper_neuron) override {
        upper_neuron_ = upper_neuron;
        upper_neuron_.AddLowerNeuron(this);
    }


    void InputNeuron::AddLowerNeuron(Neuron *lower_neuron) override {
        lower_neuron_ = lower_neuron;
        lower_neuron_.AddUpperNeuron(this);
    }


}  // mlp