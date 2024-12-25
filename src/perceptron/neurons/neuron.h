#ifndef MLP_PERCEPTRON_NEURONS_NEURON_H_
#define MLP_PERCEPTRON_NEURONS_NEURON_H_


#include <cstddef>  // size_t
#include <vector>
#include <memory>  // shared_ptr

#include "interface_neuron.h"
#include "../edge/edge.h"
// защищить по памяти перцептрон


namespace mlp {


    class Neuron : public INeuron {
        public:
            Neuron(float (*activation_function)(float&), float &k_inertia, float &step_move, std::size_t id, std::size_t layer_id);

            // void AddUpperNeuron(Neuron &upper_neuron);
            void AddLowerInChainNeuron(Neuron *lower_neuron);
            const float& GetError() const override;
            const float& GetOutput() const override;
            const std::size_t& id() const; 
            const std::size_t& layer_id() const;
            virtual void AddOutput(float value) override;
            virtual void AddChildNeuron(Neuron *child_neuron);
            virtual void UpdateWeight() override;
            virtual void UpdateChainWeight();
            virtual void UpdateAllWeight();
            virtual void ComputeOutput() override;
            virtual void ComputeChainOutput();
            virtual void ComputeAllOutput(); 
            virtual void ComputeError() override;
            virtual void ComputeChainError();
            virtual void ComputeAllError();
            virtual float GetTopCompute() override;
            virtual std::vector<float> GetAllCompute();

            Neuron* GetFirstNeuronInChain();
            Neuron* GetLastNeuronInChain();
            Neuron* GetFirstNeuronInLastLayer();
            Neuron* GetFirstNeuronInFirstLayer();
        
        protected:
            const std::vector<std::shared_ptr<Edge>>& GetParentEdges();
            const std::vector<std::shared_ptr<Edge>>& GetChildEdges();
    
        private: 
            std::size_t id_;
            std::size_t layer_id_;
            float output_{0.0};
            float error_{0.0};
            float &coefficient_of_inertia_;
            float &step_of_movement_;
            // activation_function_ - переменная с сигранутрой функции
            float (*activation_function_)(float&);

            Neuron *upper_neuron_ = nullptr;
            Neuron *lower_neuron_ = nullptr;

            std::vector<std::shared_ptr<Edge>> parent_edges_;
            std::vector<std::shared_ptr<Edge>> child_edges_;

            void AttachNeutronToParents(Neuron *neuron);
            void AttachNeutronToChildren(Neuron *neuron);
            void LinkChildNeuronWithOtherChild(Neuron *child_neuron);
            void CheckEmptyEdgesInChild(Neuron* first_in_child_neuron);
            void GetTopInChain(float &value, Neuron* neuron, Neuron* (Neuron::*shift));
            void CreateAndAddEdges(Neuron* first_neuron_in_chain, Neuron* child_neuron);
            void ComputeChain(Neuron* neuron, void (Neuron::*func)(), Neuron* (Neuron::*shift));
            Neuron* GetBoundaryNeuronInChain(Neuron* Neuron::*direction);
            Neuron* GetFirstNeuronInLayer(std::vector<std::shared_ptr<Edge>> Neuron::*edges_selector, INeuron* (Edge::*shift)() const);

    };  // Neuron


}  // mlp


#endif  // MLP_PERCEPTRON_NEURONS_NEURON_H_
