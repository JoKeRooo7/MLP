#ifndef MLP_PERCEPTRON_NEURONS_INPUT_NEURON_H_
#define MLP_PERCEPTRON_NEURONS_INPUT_NEURON_H_


#include <cstddef>  // size_t
#include <vector>


#include "template_neuron.h"
#include "../edge/edge.h"


namespace mlp {

    class InputNeuron : public Neuron {
        // public:
        //     InputNeuron();
        //     InputNeuron(std::size_t id, std::size_t layer_id);
        //     void AddOutput(float value);
        //     // void AddParrentNeuron() - not need
        //     void AddChildNeuron(Neuron *child_neuron) override;
        //     void AddUpperNeuron(Neuron *upper_neuron) override;
        //     void AddLowerNeuron(Neuron *lower_neuron) override;
        //     // void ComputeOutput();
        //     // void ComputeChainOutput();
        //     // void ComputeAllOutput();

        // private:
        //     std::size_t id_;
        //     std::size_t layer_id_;
        //     float output_{0,0};

        //     Neuron *upper_neuron_ = nullptr;
        //     Neuron *lower_neuron_ = nullptr;

        //     std::vector<Edge> parent_edges_;
        //     std::vector<Edge> child_edges_;
        //     // float error_{0,0};

    }; // InputNeuron

}  // mlp


#endif  // MLP_PERCEPTRON_NEURONS_INPUT_NEURON_H_
