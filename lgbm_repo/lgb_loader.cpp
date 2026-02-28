#include "lgb_loader.hpp"
#include <iostream>

BoosterHandle load_lgbm_model(const std::string& model_path) {
    BoosterHandle handle;
    int num_iters = 0;
    int result = LGBM_BoosterCreateFromModelfile(model_path.c_str(), &num_iters, &handle);
    
    if (result != 0) {
        std::cerr << "Failed to load LightGBM model." << std::endl;
        return nullptr;
    }
    return handle;
}
