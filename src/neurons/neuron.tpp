#include <vector>
#include <stdexcept>


#include "neuron.h"
#include "weights/weight.h"
#include "../auxiliary_modules/training_parameters.h"
#include "../auxiliary_modules/activation_function.h"

namespace mlp {

namespace graph {

template <typename T>
Neuron<T>::Neuron(std::size_t &neurol_id, std::size_t &layer_id, mlp::TrainingParameters &learning_parametrs) {
    layer_id_ = layer_id;
    neuron_id_ = neuron_id;
    learning_parametrs_ = &learning_parametrs;
}

template <typename T>
void Neuron<T>::SetType(TypeNeuron &type) {
    type_ = type;
}

template <typename T>
void Neuron<T>::SetInputValue(float &output) {
    CheckExceptNoneType();
    CheckExceptOutputType();
    CheckExceptIntermediateType();
    output_ = output;
}

template <typename T>
void Neuron<T>::ComputeOutput() {
    ExceptInputType();
    for (std::size_t i = 0; i < parents_.size(); ++i) {
        output_ += parents_[i].first.GetWeight() * parents_.second -> output_; 
    }
    output_ = mlp::sigmoid_function(output_)
}

template <typename T>
void Neuron<T>::ComputeChainOutput() {
    ExceptInputType();
    Neuron<T> *neuron = upper_;
    while (neuron != nullptr) {
        neuron -> ComputeOutput();
        neuron = neuron -> upper_;
    }
    neuron = lower_;
    while (neuron != nullptr) {
        neuron -> ComputeOutput();
        neuron = neuron -> lower_;
    }
}

template <typename T>
void Neuron<T>::ComputeAllOutput() {
    Neuron<T> *neuron = this;
    if (neuron -> type_ == TypeNeuron::Input) {
        neuron = neuron -> childs_[0].second;
    }
    while (neuron -> parent[0].second -> type_ != nullptr) {
        neuron = neuron -> parent[0].second;
    }
    do {
        neuron -> ComputeChainOutput();
        neuron = neuron -> childs_[0].second;
    } while (neuron -> type_ != TypeNeuron::Output);
}


template <typename T>
void Neuron<T>::AddUpperNeuron(Neuron<T> &other) {
    CheckOtherNeuron(other);
    if (upper_ == nullptr) {
        upper_ = &other;
        other_.lower_ = this;
    } else {
        upper_ -> lower_ = &other;
        other.upper_ = upper_;
        other.lower_ = this;
        upper_ = &other;
    }
}

template <typename T>
void Neuron<T>::AddLowerNeuron(Neuron<T> &other) {
    CheckOtherNeuron(other);
    if (lower_ == nullptr) {
        lower_ = &other;
        other_.upper_ = this; 
    } else {
        lower_ -> upper_ = &other;
        other.lower_ = lower_;
        other.upper_ = this;
        lower_ = &other;
    }
}

template <typename T>
void Neuron<T>::AddChainChildNeurons(Neuron<T> &other) {
    ExceptOutputType();
    Neuron<T> *child_neuron = SwitchingToTheUpperNeuron(other);
    for (child_neuron != nullptr) {
        // SwitchingToTheUpperNeuron(this):
        // ________________________________
        // with each iteration, we return the pointer of the first
        // connection in this layer
        child_neuron -> CreatingNetworkBetweenParent(SwitchingToTheUpperNeuron(this));
        child_neuron = child_neuron -> lower_;
    }
}

template <typename T>
void Neuron<T>::AllReconnection() {
    Neuron<T> *neuron = this;
    while (neuron -> type_ != TypeNeuron::Input) {
        CheckEmptyParentLayer(Neuron<T> *neuron);
        neuron =  neuron ->  parents_[0].second;
    }
    Neuron<T> *temp_neuron = nullptr;
    do {
        temp_neuron = neuron -> childs_[0].second;
        neuron -> childs_.clear();
        temp_neuron -> parents_.clear();
        neuron -> AddChainChildNeurons(temp_neuron);
        neuron = neuron -> childs_[0].second;
    } while (*neuron -> hilds_[0].second.type_  != TypeNeuron::Output);
}

template <typename T>
const float& Neuron<T>::GetError() {
    return error_;
}

template <typename T>
const float& Neuron<T>::GetOutputValue() {
    return output_;
}

template <typename T>
std::vector<std::pair<Weight<T>, Neuron<T>*>>& Neuron<T>::GetParentLinks() {
    ExceptInputType();
    return parents_;
}

template <typename T>
void Neuron<T>::CheckExceptNoneType() {
    if (type_ == TypeNeuron::None) {
        throw std::logic_error("You cannot perform an action in this type (None) of neuron");
    }
}

template <typename T>
void Neuron<T>::ExceptInputType() {
    if (type_ == TypeNeuron::Input) {
        throw std::logic_error("you cannot perform an action in this type (Input) of neuron");
    }
}

template <typename T>
void Neuron<T>::ExceptInputType() {
    if (type_ == TypeNeuron::Output) {
        throw std::logic_error("you cannot perform an action in this type (Output) of neuron");
    }
}

template <typename T>
void Neuron<T>::ExceptOutputType() {
    if (type_ == TypeNeuron::Intermediate) {
        throw std::logic_error("you cannot perform an action in this type (Intermediate) of neuron");
    }
}

template <typename T>
void Neuron<T>::CheckOtherNeuron(Neuron<T> &other) {
    if (other.upper_ != nullptr) {
        throw std::logic_error("You can't add a neuron from below that contains a number of other neurons from above.")
    }
    if (other.lower_ != nullptr) {
        throw std::logic_error("You can't add a neuron from above that contains a number of other neurons from below.")
    }
}

template <typename T>
void Neuron<T>::CheckEmptyChildtLayer(Neuron<T> *neuron) {
    if (neuron -> childs_.size() == 0) {
      throw std::length_error("The layer to switch to is empty, id: " +
        + std::to_string(neuron -> neurol_id_) + "layer" + std::to_string(neuron -> layer_id_));
    }
}

template <typename T>
void Neuron<T>::CheckEmptyParentLayer(Neuron<T> *neuron) {
    if (neuron -> parents_.size() == 0) {
        throw std::length_error("The layer to switch to is empty, id: " +
        + std::to_string(neuron -> neurol_id_) + "layer" + std::to_string(neuron -> layer_id_));
    }
}


template <typename T>
void Neuron<T>::CreatingNetworkBetweenParent(Neuron<T> *parrent_neuron) {
    // Adds a connection between the current neuron and the parents
    // Adds both for the current and for the parent
    for (parrent_neuron != nullptr) {
        Weight(&learning_parameters) new_weight;
        parrents_.push_pack(std::make_pair(new_weight, parrent_neuron));
        parrent_neuron -> childs_.push_back(std::make_pair(&new_weight, this));
        parrent_neuron = parrent_neuron -> lower_;
    }
}

template <typename T>
Neuron<T>& Neuron<T>::SwitchingToTheUpperNeuron(Neuron<T> &other) {
    Neuron<T>* result = &other;
    while (result -> upper_ != nullptr) {
        result = result -> upper_;
    }
    return result;
}


}  // grpah

}  // mlp
