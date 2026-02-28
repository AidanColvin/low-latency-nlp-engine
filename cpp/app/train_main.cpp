#include <iostream> // io
#include <filesystem> // dirs
#include <numeric> // iota
#include <vector> // vectors
#include <string> // strings

#include "args.h" // cli parsing
#include "folds.h" // kfold assignment
#include "tsv_reader.h" // read_train_tsv
#include "dataset.h" // build_*_dataset
#include "sgd_trainer.h" // trainer
#include "metrics.h" // accuracy
#include "model_io.h" // save/load
#include "json_write.h" // json writer

static void ensure_dir(const std::string& p) { std::filesystem::create_directories(p); } // mkdir -p

static void write_checkpoint(const std::string& out_dir, const std::string& stage, const std::string& status) { // checkpoint writer
  ensure_dir(out_dir); // ensure dir exists
  JsonWriter jw(out_dir + "/checkpoint.json"); // writer
  jw.kv("stage", stage); // stage name
  jw.kv("status", status); // ok/fail
  jw.close(); // flush
}

static std::vector<TrainRow> select_rows(const std::vector<TrainRow>& rows, const std::vector<int>& keep) { // gather subset
  std::vector<TrainRow> out; out.reserve(keep.size()); // reserve
  for (int idx : keep) out.push_back(rows[idx]); // push
  return out; // subset
}

int main(int argc, char** argv) {
  auto a = parse_args(argc, argv); // parse args

  std::string train_path = get_arg(a, "train", "data/raw/train.tsv"); // input train.tsv
  std::string out_dir = get_arg(a, "out", "outputs/04_train_cpp/run"); // output dir
  std::string mode = get_arg(a, "mode", "fit_all"); // cv or fit_all

  int dim = get_int(a, "dim", 1 << 20); // hashing dim
  int n_min = get_int(a, "n_min", 1); // min ngram
  int n_max = get_int(a, "n_max", 2); // max ngram

  int seed = get_int(a, "seed", 100); // rng seed
  int folds = get_int(a, "folds", 5); // k folds

  int epochs = get_int(a, "epochs", 8); // training epochs
  double lr = get_double(a, "lr", 0.15); // learning rate
  double l2 = get_double(a, "l2", 1.0); // L2 regularization (bigger reduces overfit)
  int patience = get_int(a, "patience", 2); // early stop patience

  ensure_dir(out_dir); // create output

  auto rows = read_train_tsv(train_path); // load rows
  int n = (int)rows.size(); // num rows

  if (mode == "cv") {
    write_checkpoint(out_dir, "cv", "running"); // stage start

    auto fold_id = make_kfold_assignments(n, folds, seed); // fold ids

    std::vector<double> fold_acc; fold_acc.reserve(folds); // store fold metrics
    for (int f = 0; f < folds; ++f) {
      std::vector<int> train_idx; train_idx.reserve(n); // indices
      std::vector<int> valid_idx; valid_idx.reserve(n / folds + 1); // indices

      for (int i = 0; i < n; ++i) { // split by fold
        if (fold_id[i] == f) valid_idx.push_back(i); else train_idx.push_back(i);
      }

      auto tr_rows = select_rows(rows, train_idx); // fold train rows
      auto va_rows = select_rows(rows, valid_idx); // fold valid rows

      Dataset tr = build_train_dataset(tr_rows, dim, n_min, n_max); // train features
      Dataset va = build_valid_dataset(va_rows, dim, n_min, n_max); // valid features

      SGDTrainer trainer; // trainer
      trainer.epochs = epochs; // epochs
      trainer.lr = lr; // lr
      trainer.l2 = l2; // l2
      trainer.patience = patience; // early stopping

      auto model = trainer.fit(tr, va); // train with validation

      auto va_hat = model.predict(va.x); // predict
      double acc = accuracy(va.y, va_hat); // compute accuracy
      fold_acc.push_back(acc); // record

      std::cout << "fold=" << f << " valid_acc=" << acc << "\n"; // log
    }

    double mean = 0.0; for (double x : fold_acc) mean += x; mean /= (double)fold_acc.size(); // mean
    double best = 0.0; for (double x : fold_acc) if (x > best) best = x; // best

    JsonWriter jw(out_dir + "/cv_metrics.json"); // metrics file
    jw.kv("train_path", train_path); // input
    jw.kv("mode", mode); // mode
    jw.kv("folds", folds); // folds
    jw.kv("seed", seed); // seed
    jw.kv("dim", dim); // dim
    jw.kv("n_min", n_min); // ngram min
    jw.kv("n_max", n_max); // ngram max
    jw.kv("epochs", epochs); // epochs
    jw.kv("lr", lr); // lr
    jw.kv("l2", l2); // l2
    jw.kv("patience", patience); // patience
    jw.kv("mean_valid_acc", mean); // mean
    jw.kv("best_valid_acc", best); // best
    jw.kv_array("fold_valid_acc", fold_acc); // per fold
    jw.close(); // flush

    write_checkpoint(out_dir, "cv", "ok"); // stage end
    std::cout << "cv_mean_valid_acc=" << mean << "\n"; // print
    return 0; // done
  }

  if (mode == "fit_all") {
    write_checkpoint(out_dir, "fit_all", "running"); // stage start

    Dataset all = build_train_dataset(rows, dim, n_min, n_max); // train on all rows

    SGDTrainer trainer; // trainer
    trainer.epochs = epochs; // epochs
    trainer.lr = lr; // lr
    trainer.l2 = l2; // l2
    trainer.patience = 0; // no early stop when training on all data

    auto model = trainer.fit_all(all); // fit on all
    save_model(out_dir + "/model.bin", model); // persist model

    write_checkpoint(out_dir, "fit_all", "ok"); // stage end
    std::cout << "wrote_model=" << out_dir + "/model.bin" << "\n"; // log
    return 0; // done
  }

  std::cerr << "unknown --mode (use cv or fit_all)\n"; // error
  write_checkpoint(out_dir, "unknown", "fail"); // checkpoint fail
  return 2; // fail
}
