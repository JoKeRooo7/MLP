#ifndef MLP_AUXILIARY_MODULES_TRAINING_PARAMETERS_H_
#define MLP_AUXILIARY_MODULES_TRAINING_PARAMETERS_H_


/* ==================================================================
This module contains the basic variables for training the model

η (step_of_movement)- is the speed of movement 
α (coefficient_of_inertia) - is the coefficient of inertia
for smoothing out sudden jumps when moving along the surface 
of the objective function.
================================================================== */

namespace mlp {


struct TrainingParameters {
    float step_of_movement = 0.0;
    float coefficient_of_inertia = 0.0;
};  // TrainingParameters


}  // mlp

#endif  // MLP_AUXILIARY_MODULES_TRAINING_PARAMETERS_H_
