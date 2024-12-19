#include "edge.h"
#include "../weight/weight.h"
#include "../neurons/interface_neuron.h"



namespace mlp {


    Edge::Edge(float &k_inertia, float &move_step) : this_weight_(k_inertia, move_step){}


    Edge::Edge(float &k_inertia, float &move_step, INeuron *left_neuron) : this_weight_(k_inertia, move_step) {
        left_neuron_ = left_neuron;
    }


    Edge::Edge(float &k_inertia, float &move_step, INeuron *left_neuron, INeuron *right_neuron) : this_weight_(k_inertia, move_step) {
        left_neuron_ = left_neuron;
        right_neuron_ = right_neuron;
    }


    void Edge::AddLeftNeuron(INeuron *left_neuron) {
        left_neuron_ = left_neuron;
    }


    void Edge::AddRightNeuron(INeuron *right_neuron) {
        right_neuron_ = right_neuron;
    }


    void Edge::UpdateWeight(float &output_, float &error_) {
        this_weight_.UpdateWeight(output_, error_);
    }


    const float& Edge::GetWeight() const {
        return this_weight_.GetWeight();
    }


    INeuron* Edge::GetLeftNeuron() const {
        return left_neuron_;
    }


    INeuron* Edge::GetRightNeuron() const {
        return right_neuron_;
    }


}  // mlp
