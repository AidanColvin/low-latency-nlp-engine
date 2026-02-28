#pragma once // fold assignment api

#include <vector> // std::vector

std::vector<int> make_kfold_assignments(int n, int k, int seed); // returns fold_id per row
