#ifndef MLP_PERCEPTRON_NEURONS_NEURON_H_
#define MLP_PERCEPTRON_NEURONS_NEURON_H_


#include <cstddef>  // size_t
#include <vector>
#include <memory>  // shared_ptr


#include "../edge/edge.h"
// защищить по памяти перцептрон


namespace mlp {


    class Neuron {
        public:
            Neuron();
            Neuron(std::size_t id, std::size_t layer_id);

            void AddUpperNeuron(Neuron *upper_neuron);
            void AddLowerNeuron(Neuron *lower_neuron);
            const float& GetError();
            const float& GetOutput();
            const std::size_t& id();
            const std::size_t& layer_id();

            virtual void AddOutput(float value);
            virtual void AddChildNeuron(Neuron *child_neuron);
            virtual void UpdateWeight();
            virtual void UpdateChainWeight();
            virtual void UpdateAllWeight();
            virtual void ComputeOutput();
            virtual void ComputeChainOutput();
            virtual void ComputeAllOutput(); 
            virtual void ComputeError();
            virtual void ComputeChainError();
            virtual void ComputeAllError();
            virtual float GetTopCompute();
            virtual std::vector<float> GetAllCompute();

            Neuron* GetFirstNeuronInChain();
            Neuron* GetFirstNeuronInLastLayer();
            Neuron* GetFirstNeuronInFirstLayer();

            // TODO UpdateWeight

        private:
            std::size_t id_;
            std::size_t layer_id_;
            float output_{0.0};
            float error_{0.0};

            Neuron *upper_neuron_ = nullptr;
            Neuron *lower_neuron_ = nullptr;

            std::vector<std::shared_ptr<Edge<Neuron>>> parent_edges_;
            std::vector<std::shared_ptr<Edge<Neuron>>> child_edges_;

            void GetTopInChain(float &value, Neuron* neuron, Neuron* (Neuron::*shift));
            void ComputeChain(Neuron* neuron, void (Neuron::*func)(), Neuron* (Neuron::*shift));
            void LinkChildNeuronWithOtherChild(Neuron *child_neuron);
            Neuron* GetFirstNeuronInLayer(Neuron* (Edge<Neuron>::*shift)());
    };  // Neuron


}  // mlp


#endif  // MLP_PERCEPTRON_NEURONS_NEURON_H_
