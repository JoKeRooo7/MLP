#ifndef MLP_GRAPH_NEURONS_NEURON_H_
#define MLP_GRAPH_NEURONS_NEURON_H_


/* ==================================================================
Class Neuron

In the Neuron class:
public:

protected fields:

protedted funcions:

private fields:

==================================================================*/

#include <vector>


#include "weights/weight.h"
#include "../auxiliary_modules/training_parameters.h"

// нужен модуль для созадния input и сбор output
namespace mlp {

namespace graph {

template <typename T>
class Neuron {
    public:
        enum TypeNeuron {
            Input,
            Intermediate,
            Output,
            None
        };

        Neuron(std::size_t &neurol_id, std::size_t &layer_id, mlp::TrainingParameters &learning_parametrs);
Í
        void SetType(TypeNeuron &type);
        void SetInputValue(float &output);
        void ComputeOutput(); 
        void ComputeChainOutput();
        void ComputeAllOutput();
        void AddUpperNeuron(Neuron<T> &other);
        void AddLowerNeuron(Neuron<T> &other);
        void AddChainChildNeurons(Neuron<T> &other);
        void CheckingForCreatedConnections();
        void AllReconnection();
        // Добавить логику обновление весов и подсчета ошибки
        // Добавить автоматическое определение типа слоя
        void RenumberingNeurons(bool change_type=false);
        const float& GetError();
        const float& GetOutputValue();
        std::vector<std::pair<Weight<T>, Neuron<T>*>>& GetLinks();

    private:
        using Numeric = T
        std::size_t layer_id_;
        std::size_t neuron_id_;

        float error_{0.0};
        float output_{0.0};

        Neuron<T> *upper_ = nullptr;
        Neuron<T> *lower_ = nullptr;
        TypeNeuron type_ = TypeNeuron::None;
        std::vector<std::pair<Weight<T>, Neuron<T>*>> parents_;
        std::vector<std::pair<Weight<T>*, Neuron<T>*>> childs_;
        mlp::TrainingParameters *learning_parametrs_ = nullptr;

        void CheckExceptNoneType();
        void CheckExceptInputType();
        void CheckExceptOutputType();
        void ConnectionBetweenChild();
        void ConnectionBetweenParent();
        void CheckExceptIntermediateType();
        void CheckOtherNeuron(Neuron<T> &other);
        void CheckEmptyChildLayer(Neuron<T> *neuron);
        void CheckEmptyParentLayer(Neuron<T> *neuron);
        void ChangeNeuronType(Neuron<T> *current_neuron);
        void CreatingNetworkBetweenParent(Neuron<T> *parrent_neuron);
        Neuron* SwitchingToTheUpperNeuron(Neuron<T> *other);
        Neuron* ReturnFirstInputNeuron(Neuron<T> *current_neuron);
        Neuron* ReturnFirstOutputNeuron(Neuron<T> *current_neuron);

        // Neuron& ReturnFirstOutputNeuron(Neuron<T> &current_neuron);
        // Neuron& ReturnLastOutputNeurom(Neuron<T> &current_neuron);
        // may be in future
        // void AddUpperNeurons(Neuron<T> &other); для вставки цепочки нейронов
        // void AddLowerNeurons(Neuron<T> &other); для вставки цепочки нейронов
        // добавить родителей (обратное ребенку)
        // void AddChainParentNeurons(Neuron<T> &other); 
        // Weight& operator=(mlp::TrainingParameters &learning_parametrs); добавить операторы

};  // Neuron

}  // graph

}  // mlp


#include "neuron.tpp"


#endif //  MLP_GRAPH_NEURONS_NEURON_H_
