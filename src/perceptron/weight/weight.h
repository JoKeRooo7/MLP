#ifndef MLP_PERCEPTRON_WEIIGHT_WEIGHT_H_
#define MLP_PERCEPTRON_WEIIGHT_WEIGHT_H_


namespace mlp {


    class Weight {
        public:
            // Weight() = default;
            Weight(float &k_inertia, float &move_step);

            // void Reset();
            void UpdateWeight(float &value, float &error);
            const float& GetWeight();

        private:
            float value_;
            float delta_prev_value_{0};
            float &coefficient_of_inertia_;
            float &step_of_movement_;

            void InitWeight();
            
    };  // Weight


}  // mlp

#endif  //  MLP_PERCEPTRON_WEIIGHT_WEIGHT_H_
