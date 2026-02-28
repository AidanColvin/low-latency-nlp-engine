#pragma once

// -------------------------------
// COMPAT SHIM: save_model / load_model
// Project defines save_model_bin/load_model_bin.
// -------------------------------
#include <string>

inline void save_model(const std::string& path, const LogRegModel& m) {
  save_model_bin(path, m);
}

inline LogRegModel load_model(const std::string& path) {
  return load_model_bin(path);
}

#include <string> // string
#include "logreg_model.h" // model

void save_model_bin(const LogRegModel& m, const std::string& path); // write model
LogRegModel load_model_bin(const std::string& path); // read model