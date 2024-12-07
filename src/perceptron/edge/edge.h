#ifndef MLP_PERCEPTRON_EDGE_EDGE_H
#define MLP_PERCEPTRON_EDGE_EDGE_H


#include "../weight/weight.h"


namespace mlp {


    template<typename N>
    class Edge {
        public:
            using Neuron = N;

            Edge() = default;
            Edge(Neuron *left_neuron);
            Edge(Neuron *left_neuron, Neuron *right_neuron);

            // void ResetWeight();
            void AddLeftNeuron(Neuron *left_neuron);
            void AddRightNeuron(Neuron *right_neuron);
            void UpdateWeight(float &output_, float &error_);
            const float& GetWeight();
            N* GetLeftNeuron();
            N* GetRightNeuron();

        private:
            Neuron *left_neuron_ = nullptr;
            Neuron *right_neuron_ = nullptr;
            Weight this_weight_;
            

    };


}  // mlp


#include "edge.tpp"


#endif  // MLP_PERCEPTRON_EDGE_EDGE_H
