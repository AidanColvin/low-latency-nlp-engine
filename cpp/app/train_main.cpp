#include <iostream>
#include <filesystem>
#include <vector>
#include <numeric>

#include "args.h"
#include "folds.h"
#include "tsv_reader.h"
#include "dataset.h"
#include "sgd_trainer.h"
#include "model_io.h"
#include "json_write.h"

static void ensure_dir(const std::string& p) { std::filesystem::create_directories(p); }

static void checkpoint(const std::string& out_dir, const std::string& stage, const std::string& status) {
  ensure_dir(out_dir);
  JsonWriter jw(out_dir + "/checkpoint.json");
  jw.kv("stage", stage);
  jw.kv("status", status);
  jw.close();
}

static std::vector<TrainRow> select_rows(const std::vector<TrainRow>& rows, const std::vector<int>& idx) {
  std::vector<TrainRow> out; out.reserve(idx.size());
  for (int i : idx) out.push_back(rows[i]);
  return out;
}

int main(int argc, char** argv) {
  auto a = parse_args(argc, argv);

  std::string mode = get_arg(a, "mode", "cv");
  std::string train_path = get_arg(a, "train", "data/raw/train.tsv");
  std::string out_dir = get_arg(a, "out", "outputs/04_train_cpp/run");

  int dim = get_int(a, "dim", 1 << 20);
  int n_min = get_int(a, "n_min", 1);
  int n_max = get_int(a, "n_max", 2);

  int folds = get_int(a, "folds", 5);
  int seed = get_int(a, "seed", 100);

  TrainConfig cfg;
  cfg.epochs = get_int(a, "epochs", 10);
  cfg.lr = (float)get_double(a, "lr", 0.12);
  cfg.l2 = (float)get_double(a, "l2", 2.0);
  cfg.thr = (float)get_double(a, "thr", 0.5);
  cfg.seed = seed;
  cfg.patience = get_int(a, "patience", 2);

  ensure_dir(out_dir);

  auto rows = read_train_tsv(train_path);
  int n = (int)rows.size();

  if (mode == "cv") {
    checkpoint(out_dir, "cv", "running");

    auto fold_id = make_kfold_assignments(n, folds, seed);

    std::vector<double> accs;
    accs.reserve(folds);

    for (int f = 0; f < folds; ++f) {
      std::vector<int> tr_idx;
      std::vector<int> va_idx;
      tr_idx.reserve(n);
      va_idx.reserve(n / folds + 1);

      for (int i = 0; i < n; ++i) {
        if (fold_id[i] == f) va_idx.push_back(i);
        else tr_idx.push_back(i);
      }

      auto tr_rows = select_rows(rows, tr_idx);
      auto va_rows = select_rows(rows, va_idx);

      Dataset tr = build_train_dataset(tr_rows, dim, n_min, n_max);
      Dataset va = build_valid_dataset(va_rows, dim, n_min, n_max);

      float train_acc = 0.0f;
      float valid_acc = 0.0f;

      auto model = train_sgd_logreg(tr, va, dim, cfg, train_acc, valid_acc);

      accs.push_back((double)valid_acc);

      std::cout << "fold=" << f << " train_acc=" << train_acc << " valid_acc=" << valid_acc << "\n";
    }

    double mean = 0.0;
    for (double x : accs) mean += x;
    mean /= (double)accs.size();

    double best = 0.0;
    for (double x : accs) if (x > best) best = x;

    JsonWriter jw(out_dir + "/cv_metrics.json");
    jw.kv("train_path", train_path);
    jw.kv("mode", mode);
    jw.kv("folds", folds);
    jw.kv("seed", seed);
    jw.kv("dim", dim);
    jw.kv("n_min", n_min);
    jw.kv("n_max", n_max);
    jw.kv("epochs", cfg.epochs);
    jw.kv("lr", (double)cfg.lr);
    jw.kv("l2", (double)cfg.l2);
    jw.kv("thr", (double)cfg.thr);
    jw.kv("patience", cfg.patience);
    jw.kv("mean_valid_acc", mean);
    jw.kv("best_valid_acc", best);
    jw.kv_array("fold_valid_acc", accs);
    jw.close();

    checkpoint(out_dir, "cv", "ok");
    std::cout << "cv_mean_valid_acc=" << mean << "\n";
    return 0;
  }

  if (mode == "fit_all") {
    checkpoint(out_dir, "fit_all", "running");

    Dataset all = build_train_dataset(rows, dim, n_min, n_max);

    float train_acc = 0.0f;
    float dummy_valid = 0.0f;
    TrainConfig cfg2 = cfg;
    cfg2.patience = 0;

    auto model = train_sgd_logreg(all, all, dim, cfg2, train_acc, dummy_valid);

    save_model(out_dir + "/model.bin", model);

    JsonWriter jw(out_dir + "/fit_metrics.json");
    jw.kv("train_path", train_path);
    jw.kv("mode", mode);
    jw.kv("dim", dim);
    jw.kv("n_min", n_min);
    jw.kv("n_max", n_max);
    jw.kv("epochs", cfg2.epochs);
    jw.kv("lr", (double)cfg2.lr);
    jw.kv("l2", (double)cfg2.l2);
    jw.kv("thr", (double)cfg2.thr);
    jw.kv("train_acc", (double)train_acc);
    jw.close();

    checkpoint(out_dir, "fit_all", "ok");
    std::cout << "wrote_model=" << out_dir + "/model.bin" << "\n";
    return 0;
  }

  checkpoint(out_dir, "unknown", "fail");
  std::cerr << "unknown --mode (use cv or fit_all)\n";
  return 2;
}
