#pragma once
#include <string> // std::string
#include <vector> // std::vector

std::vector<std::string> make_ngrams(const std::vector<std::string>& toks, int n_min, int n_max); // 1-2 or 1-3