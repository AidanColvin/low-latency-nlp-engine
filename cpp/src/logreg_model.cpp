#include "logreg_model.h" // decl
#include <cmath> // exp

LogRegModel make_model(int dim) {
  LogRegModel m; m.dim = dim; m.w.assign(dim, 0.0f); m.b = 0.0f; // zero init
  return m; // return model
}

float sigmoid(float z) {
  if (z >= 0) { float ez = std::exp(-z); return 1.0f / (1.0f + ez); } // stable
  float ez = std::exp(z); return ez / (1.0f + ez); // stable
}

float predict_proba(const LogRegModel& m, const std::vector<int>& idx, const std::vector<float>& val) {
  float z = m.b; // start with bias
  for (size_t i = 0; i < idx.size(); i++) z += m.w[idx[i]] * val[i]; // dot product
  return sigmoid(z); // probability
}

int predict_label(float p, float thr) {
  return (p >= thr) ? 1 : 0; // threshold
}