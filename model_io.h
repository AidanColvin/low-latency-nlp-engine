#pragma once
#include <string> // string
#include "logreg_model.h" // model

void save_model_bin(const LogRegModel& m, const std::string& path); // write model
LogRegModel load_model_bin(const std::string& path); // read model