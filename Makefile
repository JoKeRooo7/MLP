COMPILER = gcc
COMPILER_FLAGS = -Wall -Werror -Wextra -std=c++17
LIBS = -lstdc++ -lgtest -lgtest_main
C = $(COMPILER) $(COMPILER_FLAGS)

TEST_EDGE = tests/test_edge.cc
TEST_WEIGHT = tests/test_weight.cc
WEIGHT = src/perceptron/weight/weight.cc
NEURON = src/perceptron/neurons/neuron.cc


test: test_edge test_neuron test_weight


test_edge:
	$(C) $(WEIGHT) $(NEURON) $(TEST_EDGE)  -o test_edge.out $(LIBS)
	./test_edge.out

test_weight:
	$(C) $(WEIGHT) $(TEST_WEIGHT) -o test_weight.out $(LIBS)
	./test_weight.out


clean: clean_up_the_garbage


clean_up_the_garbage:
	find . -type f \( -name "*.out" -or -name ".DS_Store" \) -exec rm -f {} +
