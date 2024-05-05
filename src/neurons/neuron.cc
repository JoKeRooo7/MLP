#include <stdexception>


#include "neuron.h"
#include "../../support_functions/update_weight.h"


// раставить как в хедере

namespace mlp {

namespace graph {

Neuron::Neuron(std::size_t &neurol_id, std::size_t &layer_id, WeightHandler *ready_module) {
    if (ready_module == nullptr) {
        throw std::invalid_argument("the WeightModule module has not been initialized");
    }
    layer_id_ = layer_id;
    neuron_id_ = neurol_id;
}

void Neuron::set_output_value(float &output) {
    try {
        ExceptInputType();
        throw std::invalid_argument("you can't put a value outside of INPUT neuron");
    } catch (const std::logic_error &except) {
        output_value_ = output;
    }
}

float Neuron::get_output_value() {
    return output_value_;
}

Neuron::TypeNeuron Neuron::get_neuron_type() {
    return neuron_type_;
}

void Neuron::set_neuron_type(TypeNeuron type) {
    ExceptNoneType();
    type_ = type;
}

void Neuron::AddParents(std::size_t &number_of_parent, Neuron &first_parent) {
    ExceptInputType();
    Neuron *parent = first_parent;
    parents.reserve(number_of_parent);

    // нужна проверка на то, совпадают ли количество нейронов с их заявленным количеством
    while (parent != nullptr) {
        parents.emplace_back(std::make_pair(weight_tools -> InitWeight(), parent));
        parent = parent -> below;
    }
}

void Neuron::AddChilds(std::size_t &number_of_children, Neuron &first_child) {
    ExceptOutputType();
    Neuron *child = first_child;
    childrens.reserve(number_of_children);

    // нужна проверка на то, совпадают ли количество нейронов с их заявленным количеством
    while (child != nullptr) {
        // если нет веса, то надо добавить веса к этому нейрону.
        // parents.emplace_back(std::make_pair(weight_tools -> InitWeight(), parent));
        childrens.emplace_back(std::make_pair())
    } 
}

void Neuron::ExceptNoneType() {
    if (type == TypeNeuron::None) {
        throw std::logic_error("you cannot perform an action in this type (None) of neuron");
    }
}

void Neuron::ExceptInputType() {
    if (type == TypeNeuron::Input) {
        throw std::logic_error("you cannot perform an action in this type (Input) of neuron");
    }
}

void Neuron::ExceptOutputType() {
    if (type == TypeNeuron::Output) {
        throw std::logic_error("you cannot perform an action in this type (Output) of neuron");
    }
}

void Neuron::CheckExceptIntermediateType() {
    if (type == TypeNeuron::Intermediate) {
        throw std::logic_error("you cannot perform an action in this type (Intermediate) of neuron");
    }
}

}  // graph

}  // mlp
