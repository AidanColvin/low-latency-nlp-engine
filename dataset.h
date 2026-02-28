#pragma once
#include <vector> // vector
#include "sparse_vec.h" // SparseVec

struct Dataset {
  std::vector<SparseVec> x; // features
  std::vector<int> y; // labels (optional for test)
  std::vector<int> id; // ids (optional)
  int size() const { return static_cast<int>(x.size()); } // number of rows
};

Dataset build_train_dataset(const std::vector<TrainRow>& rows, int dim, int n_min, int n_max); // label+review -> features
Dataset build_valid_dataset(const std::vector<TrainRow>& rows, int dim, int n_min, int n_max); // same as train
Dataset build_test_dataset (const std::vector<TestRow>& rows,  int dim, int n_min, int n_max); // id+review -> features