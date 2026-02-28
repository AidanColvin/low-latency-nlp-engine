// train logistic regression with hashed ngrams and write model + metrics
#include "tsv_reader.h" // read train
#include "dataset.h" // build dataset
#include "sgd_trainer.h" // trainer
#include "model_io.h" // save model
#include "json_write.h" // metrics json
#include <iostream> // cout
#include <stdexcept> // runtime_error
#include <vector> // vector
#include <random> // shuffle
#include <algorithm> // shuffle

static void split_rows(const std::vector<TrainRow>& all, float train_ratio, std::vector<TrainRow>& tr, std::vector<TrainRow>& va) {
  std::vector<int> idx(all.size()); // indices
  for (int i = 0; i < (int)all.size(); i++) idx[i] = i; // fill
  std::mt19937 rng(100); // seed
  std::shuffle(idx.begin(), idx.end(), rng); // shuffle

  int n_tr = (int)(train_ratio * (float)all.size()); // train size
  tr.clear(); va.clear(); // reset
  tr.reserve(n_tr); va.reserve((int)all.size() - n_tr); // reserve

  for (int i = 0; i < (int)idx.size(); i++) {
    if (i < n_tr) tr.push_back(all[idx[i]]); else va.push_back(all[idx[i]]); // split
  }
}

int main(int argc, char** argv) {
  try {
    if (argc < 3) throw std::runtime_error("usage: train_cpp <train_tsv> <out_dir>"); // args
    std::string train_path = argv[1]; // train.tsv
    std::string out_dir = argv[2]; // outputs/04_train_cpp

    int dim = 1 << 20; // feature buckets (1,048,576)  # tune for accuracy/speed
    int n_min = 1; int n_max = 2; // unigrams + bigrams
    TrainConfig cfg{10, 0.1f, 1e-6f, 0.5f}; // epochs, lr, l2, threshold

    auto rows = read_train_tsv(train_path); // load raw rows
    std::vector<TrainRow> tr_rows, va_rows; // split holders
    split_rows(rows, 0.8f, tr_rows, va_rows); // train/valid split

    Dataset train_ds = build_train_dataset(tr_rows, dim, n_min, n_max); // hashed features
    Dataset valid_ds = build_valid_dataset(va_rows, dim, n_min, n_max); // hashed features

    float train_acc = 0.0f; float valid_acc = 0.0f; // metrics
    LogRegModel model = train_sgd_logreg(train_ds, valid_ds, dim, cfg, train_acc, valid_acc); // train

    save_model_bin(model, out_dir + "/model.bin"); // checkpoint model
    write_metrics_json(out_dir + "/metrics.json", train_acc, valid_acc); // checkpoint metrics

    std::cout << "train_acc=" << train_acc << " valid_acc=" << valid_acc << "\n"; // log
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n"; // error
    return 1;
  }
}