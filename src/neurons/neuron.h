#ifndef MLP_GRAPH_NEURONS_NEURON_H_
#define MLP_GRAPH_NEURONS_NEURON_H_


/* ==================================================================
This is an abstract class for a neuron 
containing a common implementation for derived classes

In the Neuron class:
public:

protected fields:
o (output_value_) - the output value of the neuron
TypeNeuron - all possible types of neurons

protedted funcions:
Contain two getters and a setter for setting the type of neuron.
Forbidden for rewriting in derived classes

private fields:
neuron_type_ - the field for determining which type a given neuron belongs to
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
        
        // void SetType() Добавить оператор для типа
        void SetFirstValue(float &output);
        void ComputeOutput();
        void AddUpperNeuron(Neuron<T> &other);
        void AddLowerNeuron(Neuron<T> &other);
        void AddChainChildNeurons(Neuron<T> &other);

        void AllReconnection();
    private:
        using Numeric = T
        std::size_t layer_id_;
        std::size_t neuron_id_;

        float error_{0.0};
        float output_{0.0};

        Neuron<T> upper_ = nullptr;
        Neuron<T> lower_ = nullptr;
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
        Neuron SwitchingToTheUpperNeuron(Neuron<T> &other);

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
