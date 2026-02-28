#include "ngrams.h" // declaration

std::vector<std::string> make_ngrams(const std::vector<std::string>& toks, int n_min, int n_max) {
  std::vector<std::string> out; // output
  const int T = static_cast<int>(toks.size()); // token count
  for (int i = 0; i < T; i++) { // start index
    std::string gram; // build ngram
    for (int n = 1; n <= n_max && i + n <= T; n++) { // length
      if (n == 1) gram = toks[i]; else gram += "_" + toks[i + n - 1]; // concat
      if (n >= n_min) out.push_back(gram); // store when within range
    }
  }
  return out; // ngrams
}