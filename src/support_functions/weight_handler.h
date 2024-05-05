#ifndef MLP_SUPPORT_FUNCTIONS_WEIGHT_HANDLER_H_
#define MLP_SUPPORT_FUNCTIONS_WEIGHT_HANDLER_H_


/* -----------==============================-------------
This module represents the logic of error handling and 
updating weights in the neuron class

In the WeightHandler class:

private fields:
δ (error_) - the value of the error on the neuron
w (weight) - weight between other neurons
dw (delta_weight) - weight between other neurons
η (learning_rate_) - learning rate
α (inertia_coefficient_) -  is the inertia 
coefficient to smooth out sharp overshoots 
as you move across the surface of the objective function.
--------------==============================---------- */


namespace mlp {

namespace graph {

class WeightHandler {
    public:
        WeightHandler() = default;
        ~WeightHandler() = default;
        explicit WeightHandler(const float &learning_rate,
                               const float &inertia_coefficient);

        float InitWeight();
        void InitWeight(float &weight);


        void UpdateWeigth(float &weight, 
                          float&prev_witht, 
                          float&prev_d_weight, float&error);

        float get_learning_rate();
        float get_inertia_coefficient();
    
        void set_learning_rate(const float &learning_rate);
        void set_inertia_coefficient(const float &inertia_coefficient);

    private:
        float learning_rate_;  // η
        float inertia_coefficient_; // α

};  // WeightHandler

}  // graph

}  // mlp


#endif  // MLP_SUPPORT_FUNCTIONS_WEIGHT_HANDLER_H_
