#include <random>

#include "weight.h"

namespace mlp {


    Weight::Weight(float &k_inertia, float &move_step)
     : coefficient_of_inertia_(k_inertia), step_of_movement_(move_step){
        InitWeight();
    }


    // void Weight::Reset() {
    //     InitWeitght();
    // }


    void Weight::UpdateWeight(float &value, float &error) {
        // Δwij​(n)=αΔwij​(n−1)
        delta_prev_value_ = coefficient_of_inertia_ * delta_prev_value_;
        // Δwij​(n) += (1−α)ηδj​oi​
        delta_prev_value_ += (1 - coefficient_of_inertia_) * \
        step_of_movement_ * error * value;
        // wij​(n)=wij​(n−1)−Δwij​(n),
        value_ = value_ - delta_prev_value_;
    }


    const float& Weight::GetWeight() {
        return value_;
    }


    void Weight::InitWeight() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-0.001f, 0.001f);
        value_ = dis(gen);
    }

}  // mlp
