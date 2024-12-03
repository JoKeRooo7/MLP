#include "neuron.h"
#include "../functions/activate_function.h"
#include "../edge/edge.h"
#include <cstddef>  // size_t
#include <exception>


namespace mlp {


    Neuron::Neuron() {
        std::size_t id_ = 1;
        std::size_t layer_id_ = 1;
    }


    Neuron(std::size_t id, std::size_t layer_id) {
        std::size_t id_ = id;
        std::size_t layer_id_ = layer_id;
    }


    void Neuron::AddOutput(float value) {
        output_ = value;
    }


    void Neuron::AddChildNeuron(Neuron *child_neuron) override {
        Edge new_edge;
        new_edge.AddLeftNeuron(this);
        new_edge.AddRightNeuron(child_neuron);
        child_neuron -> parent_edges_.pust_back(new_edge);
        child_edges_.push_back(new_edge);
    }


    void Neuron::AddUpperNeuron(Neuron *upper_neuron) override {
        upper_neuron_ = upper_neuron;
        upper_neuron_.AddLowerNeuron(this);
    }


    void Neuron::AddLowerNeuron(Neuron *lower_neuron) override {
        lower_neuron_ = lower_neuron;
        lower_neuron_.AddUpperNeuron(this);
    }


    void Neuron::ComputeOutput() {
        for (size_t i = 0; i < parent_edges_.size(); ++i) {
            output_ += parent_edges_[i].GetWeight() * parent_edges_[i].GetLeftOutput();
        }
        output_ = sigmoid_function(output_);
    }


    void Neuron::ComputeChainOutput() {
        Neuron *neuron = upper_;
        while (neuron != nullptr) {
            neuron -> ComputeOutput();
            neuron = neuron -> upper_;
        }
        neuron = lower_;
        while (neuron != nullptr) {
            neuron -> ComputeOutput();
            neuron = neuron -> lower_;
        }
    }


    void Neuron::ComputeAllOutput() {
        ComputeChainOutput();
        if (child_edges_.size() > 0) {
             child_edges_[0].ComputeAllOutput();
        } else {
            std::throw out_of_range("Edge not found");
        }
    }


    const float& Neuron::GetOutput() {
        return output_;
    }


    const std::size_t& Neuron::id() {
        return id;
    }


    const std::size_t& Neuron::layer_id() {
        return layer_id;
    }



}  // mlp