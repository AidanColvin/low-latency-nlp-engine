#include "metrics.h" // decl

float accuracy_dataset(const LogRegModel& m, const Dataset& ds, float thr) {
  int correct = 0; // count
  for (int i = 0; i < ds.size(); i++) {
    float p = predict_proba(m, ds.x[i].idx, ds.x[i].val); // proba
    int yhat = predict_label(p, thr); // label
    if (yhat == ds.y[i]) correct++; // match
  }
  return ds.size() ? (static_cast<float>(correct) / static_cast<float>(ds.size())) : 0.0f; // accuracy
}