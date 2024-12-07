COMPILER = gcc
COMPILER_FLAGS = -Wall -Werror -Wextra -std=c++17
LIBS = -lstdc++ -lgtest -lgtest_main
C = $(COMPILER) $(COMPILER_FLAGS)

WEIGHT = src/perceptron/weight/weight.cc
NEURON = src/perceptron/neurons/neuron.cc

test: test_edge test_neuron test_weight

test_edge:
	$(C) $(WEIGHT) $(NEURON) $ -o test_edge.out $(LIBS)
	./test_edge.out
