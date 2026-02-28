#include "folds.h" // fold assignment api
#include <random> // rng
#include <numeric> // iota
#include <algorithm> // shuffle

std::vector<int> make_kfold_assignments(int n, int k, int seed) { // deterministic fold ids
  std::vector<int> idx(n); // indices
  std::iota(idx.begin(), idx.end(), 0); // 0..n-1
  std::mt19937 rng(seed); // rng
  std::shuffle(idx.begin(), idx.end(), rng); // shuffle
  std::vector<int> fold(n, 0); // fold ids
  for (int i = 0; i < n; ++i) fold[idx[i]] = i % k; // assign by modulo
  return fold; // fold ids per row
}
