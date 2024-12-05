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


    void Neuron::AddUpperNeuron(Neuron *upper_neuron) {
        upper_neuron_ = upper_neuron;
        upper_neuron_.AddLowerNeuron(this);
    }


    void Neuron::AddLowerNeuron(Neuron *lower_neuron) {
        lower_neuron_ = lower_neuron;
        lower_neuron_.AddUpperNeuron(this);
    }


    void Neuron::ComputeOutput() { // TODO change in input neuron
        for (size_t i = 0; i < parent_edges_.size(); ++i) {
            output_ += parent_edges_[i].GetWeight() * parent_edges_[i].GetLeftOutput();
        }
        output_ = sigmoid_function(output_);
    }


    void Neuron::ComputeChainOutput() {
        ComputeChainOut(this, &Neuron::upper_);
        ComputeChainOut(lower_, &Neuron::lower_);
        // Neuron *neuron = upper_;
        // while (neuron != nullptr) {
        //     neuron -> ComputeOutput();
        //     neuron = neuron -> upper_;
        // }
        // neuron = lower_;
        // while (neuron != nullptr) {
        //     neuron -> ComputeOutput();
        //     neuron = neuron -> lower_;
        // }
    }


    void Neuron::ComputeAllOutput() { // TODO Change in output neuron
        ComputeChainOutput();
        if (child_edges_.size() > 0) {
             child_edges_[0].ComputeAllOutput();
        } else {
            std::throw out_of_range("Edge not found");
        }
    }


    const float& Neuron::GetError() {
        return error_;
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


    void Neuron::AddOutput(float value) { // TODO change in output and layer neuron
        output_ = value;
    }


    void Neuron::AddChildNeuron(Neuron *child_neuron) { // TODO change in output neuron
        Edge new_edge;
        new_edge.AddLeftNeuron(this);
        new_edge.AddRightNeuron(child_neuron);
        child_neuron -> parent_edges_.pust_back(new_edge);
        child_edges_.push_back(new_edge);
    }

    
    void Neuron::UpdateWeight() { // TODO change in input
        for (std::size_t i = 0; i < parent_edges_.size(); ++i) {
            parent_edges_[i].UpdateWeight(output_, error_);
        }
    }



    void Neuron::ComputeError() { // TODO change in output  neuron
        // δj​= oj * ​(1−oj​)​ * sun(δk * ​wjk​)
        float all_sum{0.0};
        for (std::size_t i = 0; i < child_edges_.size(); ++i) {
            all_sum += child_edges_[i].GetWeight() * child_edges_[i].GetRightError();
        }
        error_ = output_ * (1 - output_) * all_sum; 
    }


    void Neuron::ComputeChainError() { // TODO change in input  neuron
        ComputeChainErr(top_value, this, &Neuron::upper_);
        ComputeChainErr(top_value, lower_, &Neuron::lower_);
    }


    void Neuron::ComputeAllError() { // TODO change in input  neuron
        ComputeChainError();
        if (parent_edges_.size() > 0) {
            parent_edges_[0].ComputeAllError();
        } else {
            std::throw out_of_range("Edge not found");
        }
    }


    float Neuron::GetTopCompute() {
        float top_value{0.0};
        GetTopInChain(top_value, this, &Neuron::upper_);
        GetTopInChain(top_value, lower_, &Neuron::lower_);
        return top_value;
        // Neuron *neuron = upper_;
        // while (neuron != nullptr) {
        //     if (neuron -> output_ > top_value) {
        //         top_value =  neuron -> output_;
        //     }
        //     neuron = neuron -> upper_;
        // }
        // neuron = lower_;
        // while (neuron != nullptr) {
        //     if (neuron -> output_ > top_value) {
        //         top_value =  neuron -> output_; 
        //     }
        //     neuron -> ComputeOutput();
        //     neuron = neuron -> lower_;
        // }
        // return top_value;
    }

    std::vector<float> GetAllCompute() {
        std::vector<float> all_res;
        Neuron *neuron = this;
        while(neuron -> upper_ != nullptr) {
            neuron = neuron -> upper_;
        }
        while(neuron -> lower_ != nullptr) {
            all_res.push_back(neuron -> output_);
            neuron = neuron -> lower_;
        }
        return all_res;
    }


    void Neuron::ComputeChainErr(Neuron* neuron, Neuron* (Neuron::*shift)) {
        // проверить магию динамических указателей на поля класса
        while (neuron != nullptr) {
            neuron -> ComputeOutput();
            neuron = neuron->*shift;
        }
    }


    void Neuron::ComputeChainOut(Neuron* neuron, Neuron* (Neuron::*shift)) {
        // проверить магию динамических указателей на поля класса
        while (neuron != nullptr) {
            neuron -> ComputeOutput();
            neuron = neuron->*shift;
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



}  // mlp
