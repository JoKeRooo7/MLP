#include <stdlib.h> // rand()

#include "weight_habdler.h"


namespace mlp {

namespace graph {

WeightHandler::WeightHandler(const float &learning_rate,
                               const float &inertia_coefficient) {
    learning_rate_ = learning_rate;
    inertia_coefficient_ = inertia_coefficient;
}

float WeightHandler::InitWeight() {
    return ((float)std::rand() / RAND_MAX) * 0.002 - 0.001;
}

void WeightHandler::InitWeight(float &weight) {
    weight = ((float)std::rand() / RAND_MAX) * 0.002 - 0.001;
}

void WeightHandler::UpdateWeigth(float &weight, 
                                 float&prev_witht, 
                                 float&prev_d_weight, float&error) {
    float dweight, temp;
    dweight = inertia_coefficient_ * prev_d_weight;
    dweight += (1 - inertia_coefficient_)

}

float WeightHandler::get_learning_rate() {
    return learning_rate_;
}

float WeightHandler::get_inertia_coefficient() {
    return inertia_coefficient_;
}

void WeightHandler::set_learning_rate(const float &learning_rate) {
    learning_rate_ = learning_rate;
}

void WeightHandler::set_inertia_coefficient(const float &inertia_coefficient) {
    inertia_coefficient_ = inertia_coefficient;
}

}  // graph

}  // mlp
