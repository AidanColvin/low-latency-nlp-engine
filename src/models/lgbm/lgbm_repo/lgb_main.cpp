#include <iostream>
#include "lgb_config.hpp"
#include "lgb_loader.hpp"

/*
given nothing
return integer exit status
loads model and frees memory
*/
int main() {
    BoosterHandle handle = load_lgbm_model(MODEL_FILE);
    if (handle) {
        std::cout << "LightGBM model loaded successfully." << std::endl;
        LGBM_BoosterFree(handle);
    }
    return 0;
}
