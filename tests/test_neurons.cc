#include <cmath>
#include <cstddef> 
#include <gtest/gtest.h>
#include "../src/perceptron/neurons/neuron.h"

class FullTestingNeurons : public ::testing::Test {
    protected:
        static constexpr float exp = 1e-6;
        float coefficient_of_inertia_= 0.1;
        float step_of_movement_= 0.1;

};  // UnitTestingOfWeights


class TestingNeuronNetwork : public mlp::Neuron {
    public:
       using mlp::Neuron::Neuron;
       using mlp::Neuron::GetParentEdges;
       using mlp::Neuron::GetChildEdges;
};

TEST_F(FullTestingNeurons, testing_the_creation_1) {
    std::size_t id = 1, layer_id = 1;
    EXPECT_NO_THROW({
        mlp::Neuron(coefficient_of_inertia_, step_of_movement_, id, layer_id);
    });
}


TEST_F(FullTestingNeurons, testing_func_get_variables) {
    std::size_t id = 1, layer_id = 1;
    mlp::Neuron neuron(coefficient_of_inertia_, step_of_movement_, id, layer_id);
    EXPECT_NEAR(neuron.GetError(), 0.0, exp);
    EXPECT_NEAR(neuron.GetOutput(), 0.0, exp);
    EXPECT_EQ(neuron.id(), id);
    EXPECT_EQ(neuron.layer_id(), layer_id);
}


TEST_F(FullTestingNeurons, testing_func_add_output) {
    std::size_t id = 1, layer_id = 1;
    float var = 0.12345;
    mlp::Neuron neuron(coefficient_of_inertia_, step_of_movement_, id, layer_id);
    EXPECT_NO_THROW(neuron.AddOutput(var));
    EXPECT_NEAR(neuron.GetOutput(), var, exp);
}


TEST_F(FullTestingNeurons, testing_func_add_lower_neuron_1) {
    mlp::Neuron first_neuron(coefficient_of_inertia_, step_of_movement_, 1, 1);
    mlp::Neuron second_neuron(coefficient_of_inertia_, step_of_movement_, 2, 1);
    first_neuron.AddLowerNeuron(&second_neuron);
    EXPECT_EQ(second_neuron.GetFirstNeuronInChain(), &first_neuron);
}


TEST_F(FullTestingNeurons, testing_func_add_lower_neuron_2) {
    mlp::Neuron first_neuron(coefficient_of_inertia_, step_of_movement_, 1, 1);
    mlp::Neuron second_neuron(coefficient_of_inertia_, step_of_movement_, 2, 1);
    mlp::Neuron third_neuron(coefficient_of_inertia_, step_of_movement_, 3, 1);
    mlp::Neuron fourth_neuron(coefficient_of_inertia_, step_of_movement_, 4, 1);
    mlp::Neuron fifth_neuron(coefficient_of_inertia_, step_of_movement_, 5, 1);
    first_neuron.AddLowerNeuron(&second_neuron);
    second_neuron.AddLowerNeuron(&third_neuron);
    third_neuron.AddLowerNeuron(&fourth_neuron);
    fourth_neuron.AddLowerNeuron(&fifth_neuron);
    EXPECT_EQ(second_neuron.GetFirstNeuronInChain(), &first_neuron);
    EXPECT_EQ(third_neuron.GetFirstNeuronInChain(), &first_neuron);
    EXPECT_EQ(fourth_neuron.GetFirstNeuronInChain(), &first_neuron);
    EXPECT_EQ(fifth_neuron.GetFirstNeuronInChain(), &first_neuron);
}


TEST_F(FullTestingNeurons, testing_func_add_child_neuron_1) {
    mlp::Neuron first_neuron(coefficient_of_inertia_, step_of_movement_, 1, 1);
    mlp::Neuron second_neuron(coefficient_of_inertia_, step_of_movement_, 2, 1);
    mlp::Neuron first_child_neuron(coefficient_of_inertia_, step_of_movement_, 1, 2);
    first_neuron.AddLowerNeuron(&second_neuron);
    second_neuron.AddChildNeuron(&first_child_neuron);
    first_neuron.AddOutput(0.123);
    second_neuron.AddOutput(0.123);
    first_child_neuron.ComputeOutput();
    EXPECT_TRUE(std::fabs(first_child_neuron.GetOutput()) > exp);
}


TEST_F(FullTestingNeurons, testing_func_add_child_neuron_2) {
    TestingNeuronNetwork first_neuron(coefficient_of_inertia_, step_of_movement_, 1, 1);
    TestingNeuronNetwork second_neuron(coefficient_of_inertia_, step_of_movement_, 2, 1);
    TestingNeuronNetwork first_child_neuron(coefficient_of_inertia_, step_of_movement_, 1, 2);
    first_neuron.AddLowerNeuron(&second_neuron);
    first_neuron.AddChildNeuron(&first_child_neuron);
    const std::vector<std::shared_ptr<mlp::Edge>> parents_for_child = first_child_neuron.GetParentEdges();
    EXPECT_EQ(parents_for_child.size(), 2);
    EXPECT_EQ(dynamic_cast<TestingNeuronNetwork*>(parents_for_child[0] -> GetLeftNeuron()), &first_neuron);
    EXPECT_EQ(dynamic_cast<TestingNeuronNetwork*>(parents_for_child[1] -> GetLeftNeuron()), &second_neuron);
    const std::vector<std::shared_ptr<mlp::Edge>> childs_for_parent_1 = second_neuron.GetChildEdges();
    EXPECT_EQ(childs_for_parent_1.size(), 1);
    EXPECT_EQ(dynamic_cast<TestingNeuronNetwork*>(childs_for_parent_1[0] -> GetRightNeuron()), &first_child_neuron);
    const std::vector<std::shared_ptr<mlp::Edge>> childs_for_parent_2 = first_neuron.GetChildEdges();
    EXPECT_EQ(childs_for_parent_1.size(), 1);
    EXPECT_EQ(dynamic_cast<TestingNeuronNetwork*>(childs_for_parent_2[0] -> GetRightNeuron()), &first_child_neuron);
}
