#include <iostream>
#include "fasttext.h"
#include "ft_config.hpp"
#include "ft_predict.hpp"

/*
given nothing
return integer exit status
loads model and runs prediction
*/
int main() {
    fasttext::FastText model;
    model.loadModel(MODEL_FILE);
    predict_label(model, "this is a great product");
    return 0;
}
