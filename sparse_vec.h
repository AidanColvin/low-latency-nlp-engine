#pragma once
#include <vector> // vector
#include <utility> // pair
#include <algorithm> // sort

struct SparseVec {
  std::vector<int> idx; // feature indices
  std::vector<float> val; // feature values
};

SparseVec to_sparse_counts(const std::vector<int>& dims, const std::vector<int>& signs); // accumulate counts per dim