#include <cstddef>  // size_t
#include <vector>
#include <memory>  // shared_ptr
#include <stdexcept>
#include "neuron.h"
#include "../functions/activation_function.h"
#include "../edge/edge.h"



#include <iostream>


namespace mlp {


    Neuron::Neuron(float &k_inertia, float &step_move, std::size_t id, std::size_t layer_id) 
        : coefficient_of_inertia_(k_inertia), step_of_movement_(step_move)  {
        id_ = id;
        layer_id_ = layer_id;
    }


    void Neuron::AddLowerInChainNeuron(Neuron *lower_neuron) {
        Neuron *neuron = GetLastNeuronInChain();
        neuron -> lower_neuron_ = lower_neuron;
        lower_neuron -> upper_neuron_ = neuron;
        while(lower_neuron != nullptr) {
            neuron -> AttachNeutronToParents(lower_neuron);
            neuron -> AttachNeutronToChildren(lower_neuron);
            lower_neuron = lower_neuron -> lower_neuron_;
        }
    }


    const float& Neuron::GetError() const {
        return error_;
    }


    const float& Neuron::GetOutput() const {
        return output_;
    }


    const std::size_t& Neuron::id() const {
        return id_;
    }


    const std::size_t& Neuron::layer_id() const {
        return layer_id_;
    }


    void Neuron::AddOutput(float value) {  // TODO change in output and layer neuron
        output_ = value;
    }


    void Neuron::AddChildNeuron(Neuron *child_neuron) {  // TODO change in output neuron
        LinkChildNeuronWithOtherChild(child_neuron);
    }


    void Neuron::UpdateWeight() {  // TODO change in input
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
            neuron = dynamic_cast<Neuron*>(neuron -> parent_edges_[0] -> GetRightNeuron());
            neuron -> UpdateChainWeight();
        }
    }


    void Neuron::ComputeOutput() {  // TODO change in input neuron
        for (size_t i = 0; i < parent_edges_.size(); ++i) {
            output_ += parent_edges_[i] -> GetWeight() * dynamic_cast<Neuron*>(parent_edges_[i] -> GetLeftNeuron()) -> output_;
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
            current_neuron = dynamic_cast<Neuron*>(current_neuron -> child_edges_[0] -> GetRightNeuron());
            current_neuron -> ComputeChainOutput();
        }
    }


    void Neuron::ComputeError() { // TODO change in output  neuron
        // δj​= oj * ​(1−oj​)​ * sun(δk * ​wjk​)
        float all_sum{0.0};
        for (std::size_t i = 0; i < child_edges_.size(); ++i) {
            all_sum += child_edges_[i] -> GetWeight() * dynamic_cast<Neuron*>(child_edges_[i] -> GetRightNeuron()) -> GetError();
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
            neuron = dynamic_cast<Neuron*>(neuron -> parent_edges_[0] -> GetLeftNeuron());
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
        return GetBoundaryNeuronInChain(&Neuron::upper_neuron_);
    }


    Neuron* Neuron::GetLastNeuronInChain() {
        return GetBoundaryNeuronInChain(&Neuron::lower_neuron_);
    }


    Neuron* Neuron::GetFirstNeuronInLastLayer() {
        return GetFirstNeuronInLayer(&Edge::GetRightNeuron);
    }


    Neuron* Neuron::GetFirstNeuronInFirstLayer() {
        return GetFirstNeuronInLayer(&Edge::GetLeftNeuron);
    }


    const std::vector<std::shared_ptr<Edge>>& Neuron::GetParentEdges() {
        return parent_edges_;
    }


    const std::vector<std::shared_ptr<Edge>>& Neuron::GetChildEdges() {
        return child_edges_;
    }


    void Neuron::AttachNeutronToParents(Neuron *neuron) {
        // NOT_TODO добавить фунцию AttachNeuron для объединения логики
        // TODO слишком сложная логика для AttachNeuron - через ссылки, поискать другое 
        if (parent_edges_.size() != 0) {
            for (std::size_t i = 0; i < parent_edges_.size(); ++i) {
                Neuron *parent_neuron = dynamic_cast<Neuron*>(parent_edges_[i] -> GetLeftNeuron());
                std::shared_ptr<Edge> edge = std::make_shared<Edge>(coefficient_of_inertia_, step_of_movement_, parent_neuron, neuron);
                neuron -> parent_edges_.push_back(edge);
                parent_neuron -> child_edges_.push_back(edge);
            }
        }
    }


    void Neuron::AttachNeutronToChildren(Neuron *neuron) {
        if (child_edges_.size() != 0) {
            for (std::size_t i = 0; i < child_edges_.size(); ++i) {
                Neuron *child_neuron = dynamic_cast<Neuron*>(child_edges_[i] -> GetRightNeuron()) ;
                std::shared_ptr<Edge> edge = std::make_shared<Edge>(coefficient_of_inertia_, step_of_movement_, neuron, child_neuron);
                neuron -> child_edges_.push_back(edge);
                child_neuron -> parent_edges_.push_back(edge);
            }
        }
    }

    void Neuron::LinkChildNeuronWithOtherChild(Neuron *child_neuron) { 
        Neuron* first_in_child_neuron = child_neuron -> GetFirstNeuronInChain();
        Neuron* first_neuron_in_chain = GetFirstNeuronInChain();
        if (first_neuron_in_chain -> child_edges_.size() != 0) {
            dynamic_cast<Neuron*>(first_neuron_in_chain -> child_edges_[first_neuron_in_chain -> child_edges_.size() - 1] -> GetRightNeuron()) -> AddLowerInChainNeuron(child_neuron);
        } else { 
            while (first_neuron_in_chain != nullptr) {
                while (first_in_child_neuron != nullptr) {
                    std::shared_ptr<Edge> edge = std::make_shared<Edge>(coefficient_of_inertia_, step_of_movement_, first_neuron_in_chain, first_in_child_neuron);
                    first_neuron_in_chain -> child_edges_.push_back(edge);
                    first_in_child_neuron -> parent_edges_.push_back(edge);
                    first_in_child_neuron = first_in_child_neuron -> lower_neuron_;
                }
                first_in_child_neuron = child_neuron -> GetFirstNeuronInChain();
                first_neuron_in_chain = first_neuron_in_chain -> lower_neuron_;
            }
        }
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


    Neuron* Neuron::GetFirstNeuronInLayer(INeuron* (Edge::*shift)() const) { // not in output
        Neuron* neuron_in_layer = this;
        while (!neuron_in_layer->child_edges_.empty()) {
            neuron_in_layer = dynamic_cast<Neuron*>((neuron_in_layer->child_edges_[0].get() ->*shift)());
        }
        return neuron_in_layer->GetFirstNeuronInChain();
    }


    Neuron* Neuron::GetBoundaryNeuronInChain(Neuron* Neuron::*direction) {
        Neuron *current_neuron = this;
        while (current_neuron->*direction) {
            current_neuron = current_neuron->*direction;
        }
        return current_neuron;
    }


}  // mlp
