#include <string>
#include <gtest/gtest.h>
#include "../src/auxiliary_modules/training_parameters.h"
#include "../src/neurons/weights/weight.h"


class WeightTest: public ::testing::Test {
    protected:
        mlp::TrainingParameters learning_parametrs;
        mlp::graph::Weight<float> weight_obj;

        WeightTest() : weight_obj(learning_parametrs) {}

        void SetUp() override {
            learning_parametrs.step_of_movement = 0.01;
            learning_parametrs.coefficient_of_inertia = 0.9;
            weight_obj = mlp::graph::Weight<float>(learning_parametrs);
        }
};  // class WeightTests


TEST_F(WeightTest, creating_test) {
    EXPECT_THROW(mlp::graph::Weight<std::string> new_weight(learning_parametrs), std::invalid_argument);
}

TEST_F(WeightTest, update_testing) {
    const float weight_value = *weight_obj.GetWeight();
    float error{0.7}, output{1}, need_res{0.0};
    
    weight_obj.UpdateWeight(error, output);
    
    need_res = (1 - learning_parametrs.coefficient_of_inertia) * 
    learning_parametrs.step_of_movement * error * output;
    need_res = weight_value - need_res;

    // std::cout << *weight_obj.GetWeight() << std::endl;
    EXPECT_NEAR(
        *weight_obj.GetWeight(),
        need_res,
        1e-5);
}

TEST_F(WeightTest, reupdate_testing) {
    const float weight_value = *weight_obj.GetWeight();
    float error{0.7}, output{1}, delta_prev_value{0.0}, new_weight{0.0}, res{0.0};
    
    weight_obj.UpdateWeight(error, output);
    
    delta_prev_value = (1 - learning_parametrs.coefficient_of_inertia) * 
    learning_parametrs.step_of_movement * error * output;
    new_weight = weight_value - delta_prev_value;

    weight_obj.UpdateWeight(error, output);

    res = learning_parametrs.coefficient_of_inertia * delta_prev_value;
    res += (1 - learning_parametrs.coefficient_of_inertia) * 
    learning_parametrs.step_of_movement * error * output;

    res = new_weight - res;
   
    EXPECT_NEAR(
        *weight_obj.GetWeight(),
        res,
        1e-5);
}
