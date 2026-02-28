#pragma once
#include <LightGBM/c_api.h>
#include <string>

/*
given a model path string
return booster handle
loads lightgbm model via c api
*/
BoosterHandle load_lgbm_model(const std::string& model_path);
