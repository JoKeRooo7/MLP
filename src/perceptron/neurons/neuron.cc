#include "neuron.h"
#include "../functions/activation_function.h"
#include "../edge/edge.h"
#include <cstddef>  // size_t
#include <vector>
#include <memory>  // shared_ptr


namespace mlp {


    Neuron::Neuron() {
        id_ = 1;
        layer_id_ = 1;
    }


    Neuron::Neuron(std::size_t id, std::size_t layer_id) {
        id_ = id;
        layer_id_ = layer_id;
    }


    // void Neuron::AddUpperNeuron(Neuron *upper_neuron) {
    //     upper_neuron_ = upper_neuron;
    //     upper_neuron_ -> lower_neuron_ = this;
    //     ComemctChildNeorons();
    //     Вставка сверху нужна 
    // }


    void Neuron::AddLowerNeuron(Neuron *lower_neuron) {
        lower_neuron_ = lower_neuron;
        lower_neuron_ -> upper_neuron_ = this;
            //     ComemctChildNeorons();

    }

    const float& Neuron::GetError() {
        return error_;
    }


    const float& Neuron::GetOutput() {
        return output_;
    }


    const std::size_t& Neuron::id() {
        return id_;
    }


    const std::size_t& Neuron::layer_id() {
        return layer_id_;
    }

    void Neuron::AddOutput(float value) { // TODO change in output and layer neuron
        output_ = value;
    }


    void Neuron::AddChildNeuron(Neuron *child_neuron) { // TODO change in output neuron
        // пересмотреть логику а то тут песец ^._.^
        Neuron* first_neuron_in_chain = GetFirstNeuronInChain();
        first_neuron_in_chain -> LinkChildNeuronWithOtherChild(child_neuron);
        
        while (first_neuron_in_chain != nullptr) {
            std::shared_ptr<Edge<Neuron>> edge = std::make_shared<Edge<Neuron>>(first_neuron_in_chain, child_neuron);
            // edge -> AddLeftNeuron(first_neuron_in_chain);
            // edge -> AddRightNeuron(child_neuron);
            first_neuron_in_chain -> child_edges_.push_back(edge);
            child_neuron -> parent_edges_.push_back(edge);
            first_neuron_in_chain = first_neuron_in_chain -> lower_neuron_;
        }
    }


    
    void Neuron::UpdateWeight() { // TODO change in input
        for (std::size_t i = 0; i < parent_edges_.size(); ++i) {
            parent_edges_[i] -> UpdateWeight(output_, error_);
        }
    }

    
    void Neuron::UpdateChainWeight() {
        ComputeChain(this, &Neuron::UpdateWeight, &Neuron::upper_neuron_);
        ComputeChain(lower_neuron_, &Neuron::UpdateWeight, &Neuron::lower_neuron_);
    }


    void Neuron::UpdateAllWeight() {  // TODO change in output neuron 
        Neuron* neuron = GetFirstNeuronInFirstLayer();
        while (neuron -> parent_edges_.size() != 0) {
            neuron = neuron -> parent_edges_[0] -> GetRightNeuron();
            neuron -> UpdateChainWeight();
        }
    }



    void Neuron::ComputeOutput() { // TODO change in input neuron
        for (size_t i = 0; i < parent_edges_.size(); ++i) {
            output_ += parent_edges_[i] -> GetWeight() * parent_edges_[i] -> GetLeftNeuron() -> output_;
        }
        output_ = sigmoid_function(output_);
    }


    void Neuron::ComputeChainOutput() {
        ComputeChain(this, &Neuron::ComputeOutput, &Neuron::upper_neuron_);
        ComputeChain(lower_neuron_, &Neuron::ComputeOutput, &Neuron::lower_neuron_);
    }


    void Neuron::ComputeAllOutput() { // TODO Change in input neuron
        Neuron *current_neuron = GetFirstNeuronInFirstLayer();
        current_neuron -> ComputeChainOutput();
        while (child_edges_.size() > 0) {
            current_neuron = current_neuron -> child_edges_[0] -> GetRightNeuron();
            current_neuron -> ComputeChainOutput();
        }
    }


    void Neuron::ComputeError() { // TODO change in output  neuron
        // δj​= oj * ​(1−oj​)​ * sun(δk * ​wjk​)
        float all_sum{0.0};
        for (std::size_t i = 0; i < child_edges_.size(); ++i) {
            all_sum += child_edges_[i] -> GetWeight() * child_edges_[i] -> GetRightNeuron() -> GetError();
        }
        error_ = output_ * (1 - output_) * all_sum; 
    }


    void Neuron::ComputeChainError() { // TODO change in input  neuron
        ComputeChain(this, &Neuron::ComputeError, &Neuron::upper_neuron_);
        ComputeChain(lower_neuron_, &Neuron::ComputeError, &Neuron::lower_neuron_);
    }


    void Neuron::ComputeAllError() { // TODO change in input  neuron
        Neuron* neuron =  GetFirstNeuronInLastLayer();
        while (neuron -> parent_edges_.size() != 0) {
            neuron -> ComputeChainError();
            neuron = neuron -> parent_edges_[0] -> GetLeftNeuron();
        }
    }


    float Neuron::GetTopCompute() { 
        Neuron *neuron = GetFirstNeuronInLastLayer();
        float top_value{0.0};
        GetTopInChain(top_value, neuron, &Neuron::upper_neuron_);
        GetTopInChain(top_value, neuron -> lower_neuron_, &Neuron::lower_neuron_);
        return top_value;
    }

    std::vector<float> Neuron::GetAllCompute() {
        std::vector<float> all_res;
        Neuron *neuron = GetFirstNeuronInLastLayer();
        while(neuron -> lower_neuron_ != nullptr) {
            all_res.push_back(neuron -> output_);
            neuron = neuron -> lower_neuron_;
        }
        return all_res;
    }


    Neuron* Neuron::GetFirstNeuronInChain() {
        Neuron *first_neuron_in_chain = this;
        while (first_neuron_in_chain -> upper_neuron_) {
            first_neuron_in_chain = first_neuron_in_chain -> upper_neuron_;
        };
        return first_neuron_in_chain;
    }


    Neuron* Neuron::GetFirstNeuronInLastLayer() {
        return GetFirstNeuronInLayer(&Edge<Neuron>::GetRightNeuron);
    }


    Neuron* Neuron::GetFirstNeuronInFirstLayer() {
        return GetFirstNeuronInLayer(&Edge<Neuron>::GetLeftNeuron);
    }


    void Neuron::GetTopInChain(float &value, Neuron* neuron, Neuron* (Neuron::*shift)) {
        while (neuron != nullptr) {
            if (neuron -> output_ > value) {
                value =  neuron -> output_;
            }
            neuron = neuron ->*shift;
        }
    }

    
    void Neuron::ComputeChain(Neuron* neuron, void (Neuron::*func)(), Neuron* (Neuron::*shift)) {
        while (neuron != nullptr) {
            (neuron->*func)();
            neuron = neuron->*shift;
        }
    }


    void Neuron::LinkChildNeuronWithOtherChild(Neuron *child_neuron) {
        if (child_edges_.size() != 0) {
            child_edges_[child_edges_.size()] -> GetRightNeuron() -> AddLowerNeuron(child_neuron);
        }
    }


    Neuron* Neuron::GetFirstNeuronInLayer(Neuron* (Edge<Neuron>::*shift)()) { // not in output
        Neuron* neuron_in_layer = this;
        while (!neuron_in_layer->child_edges_.empty()) {
            neuron_in_layer = (neuron_in_layer->child_edges_[0].get() ->*shift)();
        }
        return neuron_in_layer->GetFirstNeuronInChain();
    }


}  // mlp
