#ifndef MLP_PERCEPTRON_NEURONS_TEMPLATE_NEURON_H_
#define MLP_PERCEPTRON_NEURONS_TEMPLATE_NEURON_H_


#include <cstddef>  // size_t
#include <vector>


namespace mlp {

    class Neuron {
        public:

        private:
            std::size_t id;
            std::size_t layer_id;
            float output_{0,0};
            
    };  // Neuron

}  // mlp

#endif  //  MLP_PERCEPTRON_NEURONS_TEMPLATE_NEURON_H_
