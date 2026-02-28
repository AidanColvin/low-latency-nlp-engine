#include "dataset.h" // decl
#include "tsv_reader.h" // TrainRow/TestRow
#include "text_clean.h" // clean_text_ascii
#include "tokenizer.h" // split_whitespace
#include "ngrams.h" // make_ngrams
#include "hashing.h" // fnv1a_64, hash_to_dim, hash_sign
#include "sparse_vec.h" // to_sparse_counts

static SparseVec featurize_review(const std::string& review, int dim, int n_min, int n_max) {
  std::string clean = clean_text_ascii(review); // normalize
  auto toks = split_whitespace(clean); // tokens
  auto feats = make_ngrams(toks, n_min, n_max); // ngrams

  std::vector<int> dims; dims.reserve(feats.size()); // hashed dims
  std::vector<int> signs; signs.reserve(feats.size()); // signed hashing

  for (auto& f : feats) {
    uint64_t h = fnv1a_64(f); // hash ngram
    dims.push_back(hash_to_dim(h, dim)); // map to bucket
    signs.push_back(hash_sign(h)); // sign
  }

  return to_sparse_counts(dims, signs); // accumulate counts -> sparse
}

Dataset build_train_dataset(const std::vector<TrainRow>& rows, int dim, int n_min, int n_max) {
  Dataset ds; ds.x.reserve(rows.size()); ds.y.reserve(rows.size()); // allocate
  for (auto& r : rows) {
    ds.x.push_back(featurize_review(r.review, dim, n_min, n_max)); // features
    ds.y.push_back(r.label); // label
  }
  return ds;
}

Dataset build_valid_dataset(const std::vector<TrainRow>& rows, int dim, int n_min, int n_max) {
  return build_train_dataset(rows, dim, n_min, n_max); // same logic
}

Dataset build_test_dataset(const std::vector<TestRow>& rows, int dim, int n_min, int n_max) {
  Dataset ds; ds.x.reserve(rows.size()); ds.id.reserve(rows.size()); // allocate
  for (auto& r : rows) {
    ds.x.push_back(featurize_review(r.review, dim, n_min, n_max)); // features
    ds.id.push_back(r.id); // id
  }
  return ds;
}