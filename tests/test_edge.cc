#include <gtest/gtest.h>

#include "../src/perceptron/edge/edge.h"
#include "../src/perceptron/neurons/neuron.h"


class UnitTestingOfEdge : public ::testing::Test {
    protected:
        float coefficient_of_inertia_= 0.1;
        float step_of_movement_= 0.1;

};  // UnitTestingOfWeights


TEST_F(UnitTestingOfEdge, testing_the_creation_1) {
    EXPECT_NO_THROW({
        mlp::Edge<mlp::Neuron> edge(coefficient_of_inertia_, step_of_movement_);
    });
}

