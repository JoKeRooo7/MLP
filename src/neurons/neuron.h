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

        void SetType(TypeNeuron &type);
        void SetInputValue(float &output);
        void ComputeOutput();
        void ComputeChainOutput();
        void ComputeAllOutput();
        void AddUpperNeuron(Neuron<T> &other);
        void AddLowerNeuron(Neuron<T> &other);
        void AddChainChildNeurons(Neuron<T> &other);
        void AllReconnection();
        const float& GetError();
        const float& GetOutputValue();
        std::vector<std::pair<Weight<T>, Neuron<T>*>>& GetLinks();

        checking for created connections
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
        void CheckExceptIntermediateType();
        void CheckOtherNeuron(Neuron<T> &other);

        void CreatingNetworkBetweenParent(Neuron<T> *parrent_neuron);
        Neuron& SwitchingToTheUpperNeuron(Neuron<T> &other);

        // debug
        void CheckEmptyChildtLayer(Neuron<T> *neuron);
        void CheckEmptyParentLayer(Neuron<T> *neuron);
        // Neuron& ReturnFirstInputNeuron(Neuron<T> &current_neuron);
        // Neuron& ReturnFirstOutputNeuron(Neuron<T> &current_neuron);
        // Neuron& ReturnLastOutputNeurom(Neuron<T> &current_neuron);
 
        // may be in future
        // void AddUpperNeurons(Neuron<T> &other); для вставки цепочки нейронов
        // void AddLowerNeurons(Neuron<T> &other); для вставки цепочки нейронов
        // void AddChainParentNeurons(Neuron<T> &other); // добавить родителей (обратное ребенку)
        // void RenumberingNeuronId(); // перенумерация
        // void UpdateAllConnection
        // Weight& operator=(mlp::TrainingParameters &learning_parametrs); добавить операторы
};  // Neuron

}  // graph

}  // mlp


#include "neuron.tpp"


#endif //  MLP_GRAPH_NEURONS_NEURON_H_
