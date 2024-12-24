#include <cmath>
#include <gtest/gtest.h>

#include "../src/perceptron/edge/edge.h"
#include "../src/perceptron/neurons/neuron.h"


class UnitTestingOfEdge : public ::testing::Test {
    protected:
        float coefficient_of_inertia_= 0.1;
        float step_of_movement_= 0.1;

};


TEST_F(UnitTestingOfEdge, testing_the_creation_1) {
    EXPECT_NO_THROW({
        mlp::Edge<mlp::Neuron> edge(coefficient_of_inertia_, step_of_movement_);
    });
}


TEST_F(UnitTestingOfEdge, testing_the_creation_2) {
    std::size_t id = 1, layer_id = 1;
    mlp::Neuron left_neuron(coefficient_of_inertia_, step_of_movement_, id, layer_id);
    EXPECT_NO_THROW({
        mlp::Edge<mlp::Neuron> edge(coefficient_of_inertia_, step_of_movement_, &left_neuron);
    });
}


TEST_F(UnitTestingOfEdge, testing_the_creation_3) {
    std::size_t id = 1, layer_id = 1;
    mlp::Neuron left_neuron(coefficient_of_inertia_, step_of_movement_, id, layer_id);
    id = layer_id = 2;
    mlp::Neuron right_neuron(coefficient_of_inertia_, step_of_movement_, id, layer_id);
    EXPECT_NO_THROW({
        mlp::Edge<mlp::Neuron> edge(coefficient_of_inertia_, step_of_movement_, &left_neuron, &right_neuron);
    });
}


TEST_F(UnitTestingOfEdge, testing_func_add_and_get_left_neuron) {
    std::size_t id = 1, layer_id = 1;
    mlp::Neuron left_neuron(coefficient_of_inertia_, step_of_movement_, id, layer_id);
    mlp::Edge<mlp::Neuron> edge(coefficient_of_inertia_, step_of_movement_);
    edge.AddLeftNeuron(&left_neuron);
    EXPECT_EQ(&left_neuron, edge.GetLeftNeuron());
}


TEST_F(UnitTestingOfEdge, testing_func_add_and_get_right_neuron) {
    std::size_t id = 1, layer_id = 1;
    mlp::Neuron right_neuron(coefficient_of_inertia_, step_of_movement_, id, layer_id);
    mlp::Edge<mlp::Neuron> edge(coefficient_of_inertia_, step_of_movement_);
    edge.AddRightNeuron(&right_neuron);
    EXPECT_EQ(&right_neuron, edge.GetRightNeuron());
}


TEST_F(UnitTestingOfEdge, testing_func_get_weight) {
    mlp::Edge<mlp::Neuron> edge(coefficient_of_inertia_, step_of_movement_);
    EXPECT_GE(edge.GetWeight(), -1.0);
    EXPECT_LE(edge.GetWeight(),  1.0);
}


TEST_F(UnitTestingOfEdge, testing_func_update_weight) {
    mlp::Edge<mlp::Neuron> edge(coefficient_of_inertia_, step_of_movement_);
    float neuron_output = 0.1f, neuron_error = 0.99f;
    float first_weight = edge.GetWeight();
    EXPECT_NO_THROW(edge.UpdateWeight(neuron_output, neuron_error));
    float second_weight = edge.GetWeight();
    EXPECT_TRUE(std::fabs(first_weight - second_weight) > 1e-6);
}


