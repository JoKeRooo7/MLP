#ifndef MLP_PERCEPTRON_NEURONS_INPUT_NEURON_H_
#define MLP_PERCEPTRON_NEURONS_INPUT_NEURON_H_


#include <cstddef>  // size_t
#include <vector>


#include "template_neuron.h"
#include "../edge/edge.h"


namespace mlp {

    class InputNeuron : public Neuron {
        public:
            InputNeuron();
            InputNeuron(std::size_t id, std::size_t layer_id);


            // add value
            // void AddChilNeuron();
            // void AddUpperNeuron(Neuron *other_neuron) override;
            // void AddLowerNeuron(Neuron *other_neuron) override;
            // void ComputeOutput(); 
            // void ComputeChainOutput();
            // void ComputeAllOutput();

        private:
            std::size_t id_;
            std::size_t layer_id_;
            float output_{0,0};

            Neuron *upper_neuron_ = nullptr;
            Neuron *lower_neuron_ = nullptr;

            std::vector<Edge> edges;
            // float error_{0,0};

    }; // InputNeuron

}  // mlp


#endif  // MLP_PERCEPTRON_NEURONS_INPUT_NEURON_H_
