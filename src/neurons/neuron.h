#ifndef MLP_GRAPH_NEURONS_NEURON_H_
#define MLP_GRAPH_NEURONS_NEURON_H_


/* -----------==============================-------------
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
--------------==============================---------- */


#include "../../support_functions/update_weight.h"


namespace mlp {

namespace graph {

class Neuron {
    public:
        Neuron() = default;
        explicit Neuron(std::size_t &neurol_id, std::size_t &layer_id, WeightHandler *ready_module);
        ~Neuron() = default;

        virtual void AddParents(std::size_t &number_of_parent, Neuron &first_parent);
        virtual void AddChilds(std::size_t &number_of_children, Neuron &first_child);
        virtual void AddAbove(Neuron first_child);
        virtual void AddBelow(Neuron first_child);

        virtual void set_output_value(float &output);
        virtual float get_output_value();

    protected:
        enum TypeNeuron {
            Input,
            Intermediate,
            Output,
            None
        };

        float output_value_{0.0};  // o

        Neuron *above{nullptr};
        Neuron *below{nullptr};

        WeightHandler *weight_tools{nullptr};

        std::vector<std::pair<float, Neuron<C>*>> parents;
        std::vector<std::pair<float*, Neuron<C>*>> childrens;


        virtual std::size_t get_layer_id();
        virtual std::size_t get_neuron_id();
        virtual TypeNeuron get_neuron_type();

        virtual void set_layer_id();
        virtual void set_neuron_id();
        virtual void set_neuron_type(TypeNeuron type);
        

    private:
        std::size_t layer_id_;
        std::size_t neuron_id_;

        TypeNeuron neuron_type_ = TypeNeuron::None;

        void CheckExceptNoneType();
        void CheckExceptInputType();
        void CheckExceptOutputType();
        void CheckExceptIntermediateType();


};  // Neuron

}  // graph

}  // mlp


#endif //  MLP_GRAPH_NEURONS_NEURON_H_
