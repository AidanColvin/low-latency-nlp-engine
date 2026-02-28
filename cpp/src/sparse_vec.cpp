#include "sparse_vec.h" // decl
#include <unordered_map> // hash map

SparseVec to_sparse_counts(const std::vector<int>& dims, const std::vector<int>& signs) {
  std::unordered_map<int, float> acc; acc.reserve(dims.size()); // accumulate
  for (size_t i = 0; i < dims.size(); i++) {
    acc[dims[i]] += static_cast<float>(signs[i]); // signed count
  }

  std::vector<std::pair<int, float>> pairs; pairs.reserve(acc.size()); // flatten
  for (auto& kv : acc) pairs.push_back(kv); // copy

  std::sort(pairs.begin(), pairs.end(), [](auto& a, auto& b){ return a.first < b.first; }); // sort by index

  SparseVec sv; sv.idx.reserve(pairs.size()); sv.val.reserve(pairs.size()); // allocate
  for (auto& p : pairs) { sv.idx.push_back(p.first); sv.val.push_back(p.second); } // store
  return sv; // sparse vector
}