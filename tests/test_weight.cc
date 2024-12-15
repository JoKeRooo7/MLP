#include <gtest/gtest.h>
#include "../src/perceptron/weight/weight.h"


class UnitTestingOfWeights : public ::testing::Test {
    protected:
        float coefficient_of_inertia_= 0.1;
        float step_of_movement_= 0.1;

};  // UnitTestingOfWeights


TEST_F(UnitTestingOfWeights, testing_the_creation) {
    EXPECT_NO_THROW({
        mlp::Weight(coefficient_of_inertia_, step_of_movement_);
    });
}


TEST_F(UnitTestingOfWeights, testing_func_get_weight) {
    mlp::Weight weight = mlp::Weight(coefficient_of_inertia_, step_of_movement_);
    EXPECT_NO_THROW({
        weight.GetWeight();
    });
}


TEST_F(UnitTestingOfWeights, testing_func_update_weigth_first_step) {
    mlp::Weight weight = mlp::Weight(coefficient_of_inertia_, step_of_movement_);
    float neuron_output = 0.73;
    float neuron_error = 0.829341;
    float init_weight = weight.GetWeight();
    EXPECT_NO_THROW({
        weight.UpdateWeight(neuron_output, neuron_error);
    });
    float d_weitght = 0 + (1 - coefficient_of_inertia_) * step_of_movement_ \
        * neuron_error * neuron_output;
    float value_weight = init_weight - d_weitght;
    EXPECT_NEAR(weight.GetWeight(), value_weight, 1e-6);
}


TEST_F(UnitTestingOfWeights, testing_func_update_weigth_two_step) {
    mlp::Weight weight = mlp::Weight(coefficient_of_inertia_, step_of_movement_);
    float neuron_output_in_first_step = 0.99;
    float neuron_error_in_first_step = 0.34215;

    float neuron_output_in_second_step = 0.33125;
    float neuron_error_in_second_step = 0.12467;

    float init_weight = weight.GetWeight();
    weight.UpdateWeight(neuron_output_in_first_step, neuron_error_in_first_step);
    float d_weitght_in_first_step = coefficient_of_inertia_ * 0 + (1 - coefficient_of_inertia_) * step_of_movement_ \
        * neuron_output_in_first_step * neuron_error_in_first_step;
    float value_weight_in_first_step = init_weight - d_weitght_in_first_step;
    EXPECT_NEAR(weight.GetWeight(), value_weight_in_first_step, 1e-6);

    weight.UpdateWeight(neuron_output_in_second_step, neuron_error_in_second_step);
    float d_weitght_in_second_step = coefficient_of_inertia_ * d_weitght_in_first_step + (1 - coefficient_of_inertia_) \
        * step_of_movement_ * neuron_output_in_second_step * neuron_error_in_second_step;
    float value_weight_in_second_step = value_weight_in_first_step - d_weitght_in_second_step;
    EXPECT_NEAR(weight.GetWeight(), value_weight_in_second_step, 1e-6);
}

