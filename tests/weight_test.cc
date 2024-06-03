#include <string>
#include <gtest/gtest.h>
#include "../src/auxiliary_modules/training_parameters.h"
#include "../src/neurons/weights/weight.h"


class Weight: public ::testing::Test {
    protected:
        mlp::TrainingParameters &learning_parametrs;
        graph::mlp::Weight<float> weight_obj;

        void SetUp() override {
            learning_parametrs.step_of_movement = 0.01;
            learning_parametrs.coefficient_of_inertia = 0.9;
            weight_obj = graph::mlp::Weight<float>();
        }
};  // class WeightTests


TEST(Weight, creating_test) {
    EXPECT_THROW(graph::mlp::Weight<std::string> new_weight, std::invalid_argument);
}

TEST_F(Weight, update_testing) {
    float *weight_value = weight_obj.GetWeight();
    float error{0.7}, output{1};
    weight_obj.UpdateWeight(&error, &output);
    EXPECT_NEAR(
        weight_obj.GetWeight(),
        weight_value - (1 - learning_parametrs.coefficient_of_inerti) * learning_parametrs.step_of_movement * error * output,
        1e-5);
}

// Добавить второй тест после первого update