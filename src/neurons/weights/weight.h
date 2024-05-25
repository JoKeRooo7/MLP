#ifndef MLP_NEURONS_WEIGHTS_H_
#define MLP_NEURONS_WEIGHTS_H_


#include <typeinfo>
#include <stdexcept>


/* -----------==============================-------------
This module is the manipulation of weights and the storage of each neuron's day weight, as well as its previous values.

In the Weight class:

private fields:
w (value_) - weight between other neurons
w_(x-1) (prev_value_) - previous weight value
dw (delta_weight) - weight between other neurons
--------------==============================---------- */


namespace mlp {

namespace graph {


template <typename T>
class Weight {
    public:
        Weight();
        ~Weight();
    
    protected:

    private:
        if constexpr (!std::is_arithmetic<T>::value) {
            throw std::invalid_argument("The type in the scale is not numerical");
        }
        using Numeric = T;

        Numeric value_;
        Numeric prev_value_;
        Numeric delta_prev_value_;
};  // Weight


}  // graph

}  // mlp


#endif  // MLP_NEURONS_WEIGHTS_H_
