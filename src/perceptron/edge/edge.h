#ifndef MLP_PERCEPTRON_EDGE_EDGE_H
#define MLP_PERCEPTRON_EDGE_EDGE_H


#include "../weight/weight.h"
#include "../neurons/interface_neuron.h"

// интерфейс N - 

namespace mlp {

    class Edge {
        public:

            Edge(float &k_inertia, float &move_step);
            Edge(float &k_inertia, float &move_step, INeuron *left_neuron);
            Edge(float &k_inertia, float &move_step, INeuron *left_neuron, INeuron *right_neuron);

            void AddLeftNeuron(INeuron *left_neuron);
            void AddRightNeuron(INeuron *right_neuron);
            void UpdateWeight(float &output_, float &error_);
            const float& GetWeight() const;
            INeuron* GetLeftNeuron() const;
            INeuron* GetRightNeuron() const;

        private:
            Weight this_weight_;
            INeuron *left_neuron_ = nullptr;
            INeuron *right_neuron_ = nullptr;
    };


}  // mlp



#endif  // MLP_PERCEPTRON_EDGE_EDGE_H
