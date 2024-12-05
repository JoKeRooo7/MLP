#ifndef MLP_PERCEPTRON_NEURONS_NEURON_H_
#define MLP_PERCEPTRON_NEURONS_NEURON_H_


#include <cstddef>  // size_t
#include <vector>


#include "template_neuron.h"
#include "../edge/edge.h"


namespace mlp {


    class Neuron {
        public:
            Neuron();
            Neuron(std::size_t id, std::size_t layer_id);

            void AddUpperNeuron(Neuron *upper_neuron);
            void AddLowerNeuron(Neuron *lower_neuron);
            // void UpdateWeight();
            // void UpdateAllWeight();
            const float& GetError();
            const float& GetOutput();
            const std::size_t& id();
            const std::size_t& layer_id();

            virtual void AddOutput(float value);
            virtual void AddChildNeuron(Neuron *child_neuron);
            virtual void UpdateWeight();
            virtual void ComputeOutput();
            virtual void ComputeChainOutput();
            virtual void ComputeAllOutput(); 
            virtual void ComputeError();
            virtual void ComputeChainError();
            virtual void ComputeAllError();
            virtual float GetTopCompute();
            virtual std::vector<float> GetAllCompute();


            // TODO UpdateWeight

        private:
            std::size_t id_;
            std::size_t layer_id_;
            float output_{0,0};
            float error_{0,0};

            Neuron *upper_neuron_ = nullptr;
            Neuron *lower_neuron_ = nullptr;

            std::vector<Edge> parent_edges_;
            std::vector<Edge> child_edges_;

            void ComputeChainErr(Neuron* neuron, Neuron* (Neuron::*shift));
            void ComputeChainOut(Neuron* neuron, Neuron* (Neuron::*shift));
            void GetTopInChain(float &value, Neuron* neuron, Neuron* (Neuron::*shift));
    };  // Neuron


}  // mlp


#endif  // MLP_PERCEPTRON_NEURONS_NEURON_H_
