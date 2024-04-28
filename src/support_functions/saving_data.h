#ifndef MLP_SUPPORT_FUNCTIONS_ACTIVATION_FUNCTION_H_
#define MLP_SUPPORT_FUNCTIONS_ACTIVATION_FUNCTION_H_


namespace mlp {


class SavingData {
    public:
        SavingData(const char *filename);
        void save_on_header();
    private:

};  // class SavingData


}  // mlp


#endif  // MLP_SUPPORT_FUNCTIONS_ACTIVATION_FUNCTION_H_
