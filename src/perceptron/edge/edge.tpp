#include "edge.h"
#include "../weight/weight.h"


namespace mlp {


    template<typename N>
    Edge<N>::Edge(Neuron *left_neuron) {
        left_neuron_ = left_neuron;
    }


    template<typename N>
    Edge<N>::Edge(Neuron *left_neuron, Neuron *right_neuron) {
        left_neuron_ = left_neuron;
        right_neuron_ = right_neuron;
    }

    // template<typename N>
    // void ResetWeight() {
    //     this_weight_.Reset();
    // }



    template<typename N>
    void Edge<N>::AddLeftNeuron(Neuron *left_neuron) {
        left_neuron_ = left_neuron;
        // ResetWeight();
    }


    template<typename N>
    void Edge<N>::AddRightNeuron(Neuron *right_neuron) {
        right_neuron_ = right_neuron;
        // ResetWeight();
    }


    template<typename N>
    void Edge<N>::UpdateWeight(float &output_, float &error_) {
        this_weight_.UpdateWeight(output_, error_);
    }


    template<typename N>
    const float& Edge<N>::GetWeight() {
        return this_weight_.GetWeight();
    }


    template<typename N>
    N* Edge<N>::GetLeftNeuron() {
        return left_neuron_;
    }


    template<typename N>
    N* Edge<N>::GetRightNeuron() {
        return right_neuron_;
    }


}  // mlp
