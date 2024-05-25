#ifndef MLP_NEURONS_WEIGHTS_H_
#define MLP_NEURONS_WEIGHTS_H_


#include <typeinfo>
#include <stdexcept>


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
        using Numeric = T

        Numeric value_;
        




};  // Weight


}  // graph

}  // mlp


#endif  // MLP_NEURONS_WEIGHTS_H_
