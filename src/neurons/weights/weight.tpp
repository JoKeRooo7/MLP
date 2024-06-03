#include <random>


#include "weight.h"


#include "../../auxiliary_modules/training_parameters.h"


namespace mlp {

namespace graph {

template <typename T>
Weight<T>::Weight(mlp::TrainingParameters &learning_parametrs) {
    *this = learning_parametrs;
}

template <typename T>
Weight<T>& Weight<T>::operator=(mlp::TrainingParameters &learning_parametrs) {
    if constexpr (!std::is_arithmetic<Numeric>::value) {
        throw std::invalid_argument("The type in the scale is not numerical");
    }
    learning_parametrs_ = &learning_parametrs;
    InitWeitght();
    return *this;
}

template <typename T>
// Добавить проверку типа для error value на численный
void Weight<T>::UpdateWeight(Numeric &error_value, Numeric &output) {
    // Δwij​(n)=αΔwij​(n−1)
    delta_prev_value_ = learning_parametrs_ -> coefficient_of_inertia * delta_prev_value_;
    // Δwij​(n) += (1−α)ηδj​oi​
    delta_prev_value_ += (1 - learning_parametrs_ -> coefficient_of_inertia) * learning_parametrs_ -> step_of_movement * error_value * output;
    // wij​(n)=wij​(n−1)−Δwij​(n),
    value_ = value_ - delta_prev_value_;
}

template <typename T>
const float *Weight<T>::GetWeight() {
    return Value();
}

template <typename T>
const float *Weight<T>::GetWeight() {
    return value_;
}

template <typename T>
void Weight<T>::InitWeitght() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dis(-0.001f, 0.001f);
    value_ = dis(gen);
}

}  // graph

}  // mlp
