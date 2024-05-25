#include <random>


#include "weight.h"


#include "../../auxiliary_modules/training_parameters.h"


namespace mlp {

namespace graph {

template <typename T>
Weight<T>::Weight(mlp::TrainingParameters &learning_parametrs) {
    *this = learning_parameters;
}

template <typename T>
Weight<T>& Weight<T>::operator=(mlp::TrainingParameters &learning_parametrs) {
    learning_parametrs_ = &learning_parametrs;
    InitWeitght();
    return *this;
}

template <typename T>
void Weight<T>::UpdateWeight(float &error_value, float &output) {
    // Δwij​(n)=αΔwij​(n−1)
    delta_prev_value_ = learning_parameters -> coefficient_of_inertia * delta_prev_value_;
    // Δwij​(n) += (1−α)ηδj​oi​
    delta_prev_value_ += (1 - learning_parameters -> coefficient_of_inertia) * learning_parameters -> step_of_movement * error_value * output;
    value_ = value_ - delta_prev_value_;
}

template <typename T>
const float *Weight<T>::GetWeight() {
    return Value()
}

template <typename T>
const float *Weight<T>::GetWeight() {
    return value_;
}

template <typename T>
Weight<T>::InitWeitght() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dis(-0.001f, 0.001f);
    value_ = dis(gen);
}

}  // graph

}  // mlp
