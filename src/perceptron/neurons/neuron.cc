#include <cstddef>  // size_t
#include <vector>
#include <memory>  // shared_ptr
#include <stdexcept>
#include "neuron.h"
#include "../edge/edge.h"



#include <iostream>


namespace mlp {


    Neuron::Neuron(float (*activation_function)(float&), float &k_inertia, float &step_move, std::size_t id, std::size_t layer_id) 
        : coefficient_of_inertia_(k_inertia), step_of_movement_(step_move), activation_function_(activation_function)   {
        id_ = id;
        layer_id_ = layer_id;
    }


    void Neuron::AddLowerInChainNeuron(Neuron *lower_neuron) {
        Neuron *neuron = GetLastNeuronInChain();
        neuron -> lower_neuron_ = lower_neuron;
        lower_neuron -> upper_neuron_ = neuron;
        while(lower_neuron != nullptr) {
            if (!lower_neuron -> parent_edges_.empty()) {
                throw std::runtime_error("the parents in the lower neuron are not empty!\n");
            }
            if (!lower_neuron -> child_edges_.empty()) {
                throw std::runtime_error("the childs in the lower neuron are not empty!\n");
            }
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
        output_ = activation_function_(output_);
    }


    void Neuron::ComputeChainOutput() {
        ComputeChain(this, &Neuron::ComputeOutput, &Neuron::upper_neuron_);
        ComputeChain(lower_neuron_, &Neuron::ComputeOutput, &Neuron::lower_neuron_);
    }


    void Neuron::ComputeAllOutput() { // TODO Change in input neuron
        Neuron *current_neuron = GetFirstNeuronInFirstLayer();
        current_neuron -> ComputeChainOutput();
        while (!current_neuron -> child_edges_.empty()) {
            current_neuron = dynamic_cast<Neuron*>(current_neuron -> child_edges_[0] -> GetRightNeuron());
            current_neuron -> ComputeChainOutput();
        }
    }


    void Neuron::ComputeError() { // TODO change in input and output neuron
        // δj​= oj * ​(1−oj​)​ * sun(δk * ​wjk​)
        float all_sum{0.0};
        for (std::size_t i = 0; i < child_edges_.size(); ++i) {
            all_sum += child_edges_[i] -> GetWeight() * dynamic_cast<Neuron*>(child_edges_[i] -> GetRightNeuron()) -> GetError();
        }
        error_ = output_ * (1 - output_) * all_sum; 
    }


    void Neuron::ComputeChainError() { // TODO change in input and output neuron
        ComputeChain(this, &Neuron::ComputeError, &Neuron::upper_neuron_);
        ComputeChain(lower_neuron_, &Neuron::ComputeError, &Neuron::lower_neuron_);
    }


    void Neuron::ComputeAllError() { // TODO change in input  neuron -> None
        Neuron* neuron =  GetFirstNeuronInLastLayer();
        while (!neuron -> parent_edges_.empty()) {
            neuron -> ComputeChainError();
            neuron = dynamic_cast<Neuron*>(neuron -> parent_edges_[0] -> GetLeftNeuron());
        }
    }


    float Neuron::GetTopOutput() { 
        Neuron *neuron = GetFirstNeuronInLastLayer();
        float top_value{0.0};
        GetTopInChain(top_value, neuron, &Neuron::upper_neuron_);
        GetTopInChain(top_value, neuron -> lower_neuron_, &Neuron::lower_neuron_);
        return top_value;
    }


    std::vector<float> Neuron::GetAllOutput() {
        std::vector<float> all_res;
        Neuron *neuron = GetFirstNeuronInLastLayer();
        while(neuron != nullptr) {
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
        return GetFirstNeuronInLayer(&Neuron::child_edges_, &Edge::GetRightNeuron);
    }


    Neuron* Neuron::GetFirstNeuronInFirstLayer() {
        return GetFirstNeuronInLayer(&Neuron::parent_edges_, &Edge::GetLeftNeuron);
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
        Neuron* first_neuron_in_chain = GetFirstNeuronInChain();
        CheckEmptyEdgesInChild(child_neuron -> GetFirstNeuronInChain());
        if (first_neuron_in_chain -> child_edges_.size() != 0) {
            dynamic_cast<Neuron*>(first_neuron_in_chain -> child_edges_[first_neuron_in_chain -> child_edges_.size() - 1] -> GetRightNeuron()) -> AddLowerInChainNeuron(child_neuron);
        } else { 
            CreateAndAddEdges(first_neuron_in_chain, child_neuron);
        }
    }


    void Neuron::CheckEmptyEdgesInChild(Neuron* first_in_child_neuron) {
        while (first_in_child_neuron != nullptr) {
            if (!first_in_child_neuron->parent_edges_.empty()) {
                throw std::runtime_error("The parents in the child neuron are not empty!\n");
            }
            if (!first_in_child_neuron->child_edges_.empty()) {
                throw std::runtime_error("The childs in the child neuron are not empty!\n");
            }
            first_in_child_neuron = first_in_child_neuron->lower_neuron_;
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


    void Neuron::CreateAndAddEdges(Neuron* first_neuron_in_chain, Neuron* child_neuron) {
        while (first_neuron_in_chain != nullptr) {
            Neuron* first_in_child_neuron = child_neuron->GetFirstNeuronInChain();
            while (first_in_child_neuron != nullptr) {
                std::shared_ptr<Edge> edge = std::make_shared<Edge>(coefficient_of_inertia_, step_of_movement_, first_neuron_in_chain, first_in_child_neuron);
                first_neuron_in_chain->child_edges_.push_back(edge);
                first_in_child_neuron->parent_edges_.push_back(edge);
                first_in_child_neuron = first_in_child_neuron->lower_neuron_;
            }
            first_neuron_in_chain = first_neuron_in_chain->lower_neuron_;
        }
    }
    

    void Neuron::ComputeChain(Neuron* neuron, void (Neuron::*func)(), Neuron* (Neuron::*shift)) {
        while (neuron != nullptr) {
            (neuron->*func)();
            neuron = neuron->*shift;
        }
    }



    Neuron* Neuron::GetBoundaryNeuronInChain(Neuron* Neuron::*direction) {
        Neuron *current_neuron = this;
        while (current_neuron->*direction) {
            current_neuron = current_neuron->*direction;
        }
        return current_neuron;
    }


    Neuron* Neuron::GetFirstNeuronInLayer(
        std::vector<std::shared_ptr<Edge>> Neuron::*edges_selector,
        INeuron* (Edge::*shift)() const) {
        
        Neuron* neuron_in_layer = this;
        std::vector<std::shared_ptr<Edge>> edges = neuron_in_layer->*edges_selector;
        while (!edges.empty()) {
            neuron_in_layer = dynamic_cast<Neuron*>((edges[0].get()->*shift)());
            edges = neuron_in_layer->*edges_selector;
        }
        return neuron_in_layer -> GetFirstNeuronInChain();
    }

}  // mlp
