#include <cmath>
#include <cstddef> 
#include <gtest/gtest.h>
#include "../src/perceptron/edge/edge.h"
#include "../src/perceptron/neurons/neuron.h"

#include <iostream>


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

};  // TestingNeuronNetwork



class FullNeuralNetworkTesting : public FullTestingNeurons {
    protected:
        TestingNeuronNetwork first_neuron, second_neuron;
        TestingNeuronNetwork first_child_neuron, second_child_neuron, third_child_neuron;
        TestingNeuronNetwork first_last_neuron, second_last_neuron;
        FullNeuralNetworkTesting()
            : first_neuron(coefficient_of_inertia_, step_of_movement_, 1, 1),
            second_neuron(coefficient_of_inertia_, step_of_movement_, 2, 1),
            first_child_neuron(coefficient_of_inertia_, step_of_movement_, 1, 2),
            second_child_neuron(coefficient_of_inertia_, step_of_movement_, 2, 2),
            third_child_neuron(coefficient_of_inertia_, step_of_movement_, 3, 2),
            first_last_neuron(coefficient_of_inertia_, step_of_movement_, 1, 3),
            second_last_neuron(coefficient_of_inertia_, step_of_movement_, 2, 3) {
                first_neuron.AddLowerInChainNeuron(&second_neuron);
                first_neuron.AddChildNeuron(&first_child_neuron);
                first_neuron.AddChildNeuron(&second_child_neuron);
                first_neuron.AddChildNeuron(&third_child_neuron);
                first_child_neuron.AddChildNeuron(&first_last_neuron);
                first_child_neuron.AddChildNeuron(&second_last_neuron);
            }   

    void SetUp() override {}
};  // FullNeuralNetworkTesting




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
    first_neuron.AddLowerInChainNeuron(&second_neuron);
    EXPECT_EQ(second_neuron.GetFirstNeuronInChain(), &first_neuron);
}


TEST_F(FullTestingNeurons, testing_func_add_lower_neuron_2) {
    mlp::Neuron first_neuron(coefficient_of_inertia_, step_of_movement_, 1, 1);
    mlp::Neuron second_neuron(coefficient_of_inertia_, step_of_movement_, 2, 1);
    mlp::Neuron third_neuron(coefficient_of_inertia_, step_of_movement_, 3, 1);
    mlp::Neuron fourth_neuron(coefficient_of_inertia_, step_of_movement_, 4, 1);
    mlp::Neuron fifth_neuron(coefficient_of_inertia_, step_of_movement_, 5, 1);
    first_neuron.AddLowerInChainNeuron(&second_neuron);
    second_neuron.AddLowerInChainNeuron(&third_neuron);
    third_neuron.AddLowerInChainNeuron(&fourth_neuron);
    fourth_neuron.AddLowerInChainNeuron(&fifth_neuron);
    EXPECT_EQ(second_neuron.GetFirstNeuronInChain(), &first_neuron);
    EXPECT_EQ(third_neuron.GetFirstNeuronInChain(), &first_neuron);
    EXPECT_EQ(fourth_neuron.GetFirstNeuronInChain(), &first_neuron);
    EXPECT_EQ(fifth_neuron.GetFirstNeuronInChain(), &first_neuron);
}


TEST_F(FullTestingNeurons, testing_func_add_child_neuron_1) {
    mlp::Neuron first_neuron(coefficient_of_inertia_, step_of_movement_, 1, 1);
    mlp::Neuron second_neuron(coefficient_of_inertia_, step_of_movement_, 2, 1);
    mlp::Neuron first_child_neuron(coefficient_of_inertia_, step_of_movement_, 1, 2);
    first_neuron.AddLowerInChainNeuron(&second_neuron);
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
    first_neuron.AddLowerInChainNeuron(&second_neuron);
    first_neuron.AddChildNeuron(&first_child_neuron);
    const std::vector<std::shared_ptr<mlp::Edge>> parents_for_child = first_child_neuron.GetParentEdges();
    EXPECT_EQ(parents_for_child.size(), 2);
    EXPECT_EQ(dynamic_cast<TestingNeuronNetwork*>(parents_for_child[0] -> GetLeftNeuron()), &first_neuron);
    EXPECT_EQ(dynamic_cast<TestingNeuronNetwork*>(parents_for_child[1] -> GetLeftNeuron()), &second_neuron);
    const std::vector<std::shared_ptr<mlp::Edge>> childs_for_parent_1 = second_neuron.GetChildEdges();
    EXPECT_EQ(childs_for_parent_1.size(), 1);
    EXPECT_EQ(dynamic_cast<TestingNeuronNetwork*>(childs_for_parent_1[0] -> GetRightNeuron()), &first_child_neuron);
    const std::vector<std::shared_ptr<mlp::Edge>> childs_for_parent_2 = first_neuron.GetChildEdges();
    EXPECT_EQ(childs_for_parent_2.size(), 1);
    EXPECT_EQ(dynamic_cast<TestingNeuronNetwork*>(childs_for_parent_2[0] -> GetRightNeuron()), &first_child_neuron);
}


TEST_F(FullTestingNeurons, testing_func_add_child_neuron_3) {
    TestingNeuronNetwork first_neuron(coefficient_of_inertia_, step_of_movement_, 1, 1);
    TestingNeuronNetwork second_neuron(coefficient_of_inertia_, step_of_movement_, 2, 1);
    TestingNeuronNetwork first_child_neuron(coefficient_of_inertia_, step_of_movement_, 1, 2);
    TestingNeuronNetwork second_child_neuron(coefficient_of_inertia_, step_of_movement_, 1, 2);
    first_neuron.AddLowerInChainNeuron(&second_neuron);
    first_neuron.AddChildNeuron(&first_child_neuron);
    first_neuron.AddChildNeuron(&second_child_neuron);
    const std::vector<std::shared_ptr<mlp::Edge>> parents_for_child_1 = first_child_neuron.GetParentEdges();
    EXPECT_EQ(parents_for_child_1.size(), 2);
    EXPECT_EQ(dynamic_cast<TestingNeuronNetwork*>(parents_for_child_1[0] -> GetLeftNeuron()), &first_neuron);
    EXPECT_EQ(dynamic_cast<TestingNeuronNetwork*>(parents_for_child_1[1] -> GetLeftNeuron()), &second_neuron);
    const std::vector<std::shared_ptr<mlp::Edge>> parents_for_child_2= second_child_neuron.GetParentEdges();
    EXPECT_EQ(parents_for_child_2.size(), 2);
    EXPECT_EQ(dynamic_cast<TestingNeuronNetwork*>(parents_for_child_2[0] -> GetLeftNeuron()), &first_neuron);
    EXPECT_EQ(dynamic_cast<TestingNeuronNetwork*>(parents_for_child_2[1] -> GetLeftNeuron()), &second_neuron);
    const std::vector<std::shared_ptr<mlp::Edge>> childs_for_parent_1 = second_neuron.GetChildEdges();
    EXPECT_EQ(childs_for_parent_1.size(), 2);
    EXPECT_EQ(dynamic_cast<TestingNeuronNetwork*>(childs_for_parent_1[0] -> GetRightNeuron()), &first_child_neuron);
    EXPECT_EQ(dynamic_cast<TestingNeuronNetwork*>(childs_for_parent_1[1] -> GetRightNeuron()), &second_child_neuron);
    const std::vector<std::shared_ptr<mlp::Edge>> childs_for_parent_2 = first_neuron.GetChildEdges();
    EXPECT_EQ(childs_for_parent_2.size(), 2);
    EXPECT_EQ(dynamic_cast<TestingNeuronNetwork*>(childs_for_parent_1[0] -> GetRightNeuron()), &first_child_neuron);
    EXPECT_EQ(dynamic_cast<TestingNeuronNetwork*>(childs_for_parent_1[1] -> GetRightNeuron()), &second_child_neuron);
}


TEST_F(FullTestingNeurons, testing_func_add_child_neuron_4) {
    TestingNeuronNetwork first_neuron(coefficient_of_inertia_, step_of_movement_, 1, 1);
    TestingNeuronNetwork second_neuron(coefficient_of_inertia_, step_of_movement_, 2, 1);
    TestingNeuronNetwork first_child_neuron(coefficient_of_inertia_, step_of_movement_, 1, 2);
    TestingNeuronNetwork second_child_neuron(coefficient_of_inertia_, step_of_movement_, 1, 2);
    first_neuron.AddLowerInChainNeuron(&second_neuron);
    first_child_neuron.AddLowerInChainNeuron(&second_child_neuron);
    first_neuron.AddChildNeuron(&second_child_neuron);
    const std::vector<std::shared_ptr<mlp::Edge>> parents_for_child_1 = first_child_neuron.GetParentEdges();
    EXPECT_EQ(parents_for_child_1.size(), 2);
    EXPECT_EQ(dynamic_cast<TestingNeuronNetwork*>(parents_for_child_1[0] -> GetLeftNeuron()), &first_neuron);
    EXPECT_EQ(dynamic_cast<TestingNeuronNetwork*>(parents_for_child_1[1] -> GetLeftNeuron()), &second_neuron);
    const std::vector<std::shared_ptr<mlp::Edge>> parents_for_child_2= second_child_neuron.GetParentEdges();
    EXPECT_EQ(parents_for_child_2.size(), 2);
    EXPECT_EQ(dynamic_cast<TestingNeuronNetwork*>(parents_for_child_2[0] -> GetLeftNeuron()), &first_neuron);
    EXPECT_EQ(dynamic_cast<TestingNeuronNetwork*>(parents_for_child_2[1] -> GetLeftNeuron()), &second_neuron);
    const std::vector<std::shared_ptr<mlp::Edge>> childs_for_parent_1 = second_neuron.GetChildEdges();
    EXPECT_EQ(childs_for_parent_1.size(), 2);
    EXPECT_EQ(dynamic_cast<TestingNeuronNetwork*>(childs_for_parent_1[0] -> GetRightNeuron()), &first_child_neuron);
    EXPECT_EQ(dynamic_cast<TestingNeuronNetwork*>(childs_for_parent_1[1] -> GetRightNeuron()), &second_child_neuron);
    const std::vector<std::shared_ptr<mlp::Edge>> childs_for_parent_2 = first_neuron.GetChildEdges();
    EXPECT_EQ(childs_for_parent_2.size(), 2);
    EXPECT_EQ(dynamic_cast<TestingNeuronNetwork*>(childs_for_parent_1[0] -> GetRightNeuron()), &first_child_neuron);
    EXPECT_EQ(dynamic_cast<TestingNeuronNetwork*>(childs_for_parent_1[1] -> GetRightNeuron()), &second_child_neuron);
}


TEST_F(FullTestingNeurons, testing_add_get_output) {
    TestingNeuronNetwork first_neuron(coefficient_of_inertia_, step_of_movement_, 1, 1);
    TestingNeuronNetwork second_neuron(coefficient_of_inertia_, step_of_movement_, 2, 1);
    TestingNeuronNetwork first_child_neuron(coefficient_of_inertia_, step_of_movement_, 1, 2);
    first_neuron.AddLowerInChainNeuron(&second_neuron);
    second_neuron.AddChildNeuron(&first_child_neuron);
    float first_value = 0.1;
    float second_value = 0.125;
    float first_child_value = 0.22;
    EXPECT_NO_THROW({
        first_neuron.AddOutput(first_value);
    });
    EXPECT_NO_THROW({
        second_neuron.AddOutput(second_value);
    });
    EXPECT_NO_THROW({
        first_child_neuron.AddOutput(first_child_value);
    });
    EXPECT_NEAR(first_neuron.GetOutput(), first_value, exp);
    EXPECT_NEAR(second_neuron.GetOutput(), second_value, exp);
    EXPECT_NEAR(first_child_neuron.GetOutput(), first_child_value, exp);
}


TEST_F(FullNeuralNetworkTesting, testing_update_weight) {
    const std::vector<std::shared_ptr<mlp::Edge>>& neuron_edge = first_child_neuron.GetParentEdges();
    std::shared_ptr<mlp::Edge> his_first_neuron = neuron_edge[0];
    std::shared_ptr<mlp::Edge> his_second_neuron = neuron_edge[1];
    float pre_first_value = his_first_neuron -> GetWeight();
    float pre_second_value = his_second_neuron -> GetWeight();
    EXPECT_NO_THROW({
        first_child_neuron.UpdateWeight();
    });
    float new_first_weight, new_second_weight = 0.0;
    new_first_weight = coefficient_of_inertia_ * 0 + (1 - coefficient_of_inertia_) * \
        step_of_movement_ * first_neuron.GetOutput() * first_neuron.GetError();
    new_first_weight = pre_first_value - new_first_weight;
    new_second_weight = coefficient_of_inertia_ * 0 + (1 - coefficient_of_inertia_) * \
        step_of_movement_ * second_neuron.GetOutput() * second_neuron.GetError();
    new_second_weight = pre_second_value - new_second_weight;
    EXPECT_NEAR(new_first_weight, neuron_edge[0] -> GetWeight(), exp);
    EXPECT_NEAR(new_second_weight, neuron_edge[1] -> GetWeight(), exp);
}


TEST_F(FullNeuralNetworkTesting, testing_update_chain_and_all) {
    EXPECT_NO_THROW(first_neuron.UpdateChainWeight());
    first_neuron.AddOutput(0.2);
    second_neuron.AddOutput(0.3);
    EXPECT_NO_THROW(first_neuron.UpdateChainWeight());
    const std::vector<std::shared_ptr<mlp::Edge>>& neuron_edge = first_child_neuron.GetParentEdges();
    std::shared_ptr<mlp::Edge> his_first_neuron = neuron_edge[0];
    std::shared_ptr<mlp::Edge> his_second_neuron = neuron_edge[1];
    float pre_first_value = his_first_neuron -> GetWeight();
    float pre_second_value = his_second_neuron -> GetWeight();
    float new_first_weight, new_second_weight = 0.0;
    new_first_weight = coefficient_of_inertia_ * 0 + (1 - coefficient_of_inertia_) * \
        step_of_movement_ * first_neuron.GetOutput() * first_neuron.GetError();
    new_first_weight = pre_first_value - new_first_weight;
    new_second_weight = coefficient_of_inertia_ * 0 + (1 - coefficient_of_inertia_) * \
        step_of_movement_ * second_neuron.GetOutput() * second_neuron.GetError();
    new_second_weight = pre_second_value - new_second_weight;
    EXPECT_NEAR(new_first_weight, neuron_edge[0] -> GetWeight(), exp);
    EXPECT_NEAR(new_second_weight, neuron_edge[1] -> GetWeight(), exp);
}


// TODO tests for
// virtual void ComputeOutput() override;
// virtual void ComputeChainOutput();
// virtual void ComputeAllOutput(); 
// virtual void ComputeError() override;
// virtual void ComputeChainError();
// virtual void ComputeAllError();
// virtual float GetTopCompute() override;
// virtual std::vector<float> GetAllCompute();
// Neuron* GetFirstNeuronInChain();
// Neuron* GetFirstNeuronInLastLayer();
// Neuron* GetFirstNeuronInFirstLayer();
