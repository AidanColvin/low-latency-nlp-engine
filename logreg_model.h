#pragma once
#include <vector> // vector

struct LogRegModel {
  int dim; // feature dimension
  std::vector<float> w; // weights
  float b; // bias
};

LogRegModel make_model(int dim); // init weights
float sigmoid(float z); // sigmoid
float predict_proba(const LogRegModel& m, const std::vector<int>& idx, const std::vector<float>& val); // p(y=1)
int predict_label(float p, float thr); // 0/1