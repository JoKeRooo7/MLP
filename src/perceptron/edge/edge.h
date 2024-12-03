#ifndef MLP_PERCEPTRON_EDGE_EDGE_H
#define MLP_PERCEPTRON_EDGE_EDGE_H


#include "../neurons/template_neuron.h"


namespace mlp {

    template <typename W>
    class Edge {
        public:
            using Weight = W;

            Edge() = default;
            Edge(Neuron *left_neuron);
            Edge(Neuron *left_neuron, Neuron *right_neuron);
            void AddLeftNeuron(Neuron *left_neuron);
            void AddRightNeuron(Neufon *right_neuron);
            void ComputeAllOutput();
            const float& GetWeight();
            const float&  GetLeftOutput();
            const float&  GetRightOutput();
            // TODO - add UPDATE WEIGHT
            // TODO - add calculcate error
            // TODO -  add lear
            ~Edge();
        
        private:
            Neuron *left_neuron_ = nullptr;
            Neuron *right_neuron_ = nullptr;
            Weight this_weight_;

            // void ResetWeight();
    };


}  // mlp


#endif  // MLP_PERCEPTRON_EDGE_EDGE_H
