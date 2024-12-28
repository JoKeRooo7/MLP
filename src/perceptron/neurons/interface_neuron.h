#ifndef MLP_PERCEPTRON_INTERFACE_NEURON_H_
#define MLP_PERCEPTRON_INTERFACE_NEURON_H_


#include <cstddef>
#include <vector>


namespace mlp {


    class INeuron {
    public:
        virtual ~INeuron() = default;

        virtual void AddOutput(float value) = 0;
        virtual void UpdateWeight() = 0;
        virtual void ComputeOutput() = 0;
        virtual void ComputeError() = 0;
        virtual float GetTopOutput() = 0;
        virtual const float& GetError() const = 0;
        virtual const float& GetOutput() const = 0;
    };


}  // mlp

#endif  // MLP_PERCEPTRON_INTERFACE_NEURON_H_

