#include <random>


#include "weight.h"


#include <iostream>

#include "../../auxiliary_modules/training_parameters.h"


namespace mlp {

namespace graph {

template <typename T>
Weight<T>::Weight(mlp::TrainingParameters &learning_parametrs) {
    *this = learning_parametrs;
}

template <typename T>
Weight<T>& Weight<T>::operator=(mlp::TrainingParameters& learning_parametrs) {
    CheckNumericType<T>();
    learning_parametrs_ = &learning_parametrs;
    InitWeitght();
    return *this;
}

template <typename T>
template <typename X,typename Y>
void Weight<T>::UpdateWeight(X& error_value, Y& output) {
    CheckNumericType<X>();
    CheckNumericType<Y>();

    // Δwij​(n)=αΔwij​(n−1)
    delta_prev_value_ = learning_parametrs_ -> coefficient_of_inertia * delta_prev_value_;
    // Δwij​(n) += (1−α)ηδj​oi​
    delta_prev_value_ += (1 - learning_parametrs_ -> coefficient_of_inertia) * 
    learning_parametrs_ -> step_of_movement * error_value * output;
    // wij​(n)=wij​(n−1)−Δwij​(n),
    value_ = value_ - delta_prev_value_;
}

template <typename T>
const T* Weight<T>::GetWeight() {
    return Value();
}

template <typename T>
const T* Weight<T>::Value() {
    return &value_;
}

template <typename T>
void Weight<T>::InitWeitght() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-0.001f, 0.001f);
    value_ = dis(gen);
}

template <typename T>
template <typename C>
void Weight<T>::CheckNumericType() {
    if constexpr (!std::is_arithmetic<C>::value) {
        throw std::invalid_argument("The type in the scale is not numerical");
    }
}

}  // namespace graph

}  // namespace mlp
