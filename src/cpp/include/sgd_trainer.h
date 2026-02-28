#pragma once
#include "logreg_model.h" // model
#include "dataset.h" // dataset

struct TrainConfig {
  int seed; // shuffle seed
  int patience; // early stop patience

  int epochs; // passes over data
  float lr; // learning rate
  float l2; // L2 regularization
  float thr; // prediction threshold
};

LogRegModel train_sgd_logreg(const Dataset& train, const Dataset& valid, int dim, const TrainConfig& cfg,
                            float& out_train_acc, float& out_valid_acc); // train and return model