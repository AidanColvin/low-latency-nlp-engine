#include "sgd_trainer.h" // decl
#include "metrics.h" // accuracy
#include <random> // shuffle rng
#include <numeric> // iota
#include <algorithm> // shuffle
#include <limits> // lowest

LogRegModel train_sgd_logreg(const Dataset& train, const Dataset& valid, int dim, const TrainConfig& cfg,
                            float& out_train_acc, float& out_valid_acc) {
  LogRegModel model = make_model(dim); // init model

  std::vector<int> order(train.size()); // indices
  std::iota(order.begin(), order.end(), 0); // 0..n-1

  std::mt19937 rng(cfg.seed); // deterministic shuffle

  float best_valid = std::numeric_limits<float>::lowest(); // best valid acc
  int bad = 0; // patience counter

  for (int ep = 0; ep < cfg.epochs; ep++) {
    std::shuffle(order.begin(), order.end(), rng); // random order each epoch

    for (int ii : order) {
      const auto& x = train.x[ii]; // sparse features
      int y = train.y[ii]; // label

      float p = predict_proba(model, x.idx, x.val); // predicted probability
      float g = (p - static_cast<float>(y)); // gradient of log loss wrt z

      // weights update: w = w - lr * (g*x + l2*w)
      for (size_t k = 0; k < x.idx.size(); k++) {
        int j = x.idx[k]; // feature index
        float xj = x.val[k]; // feature value
        model.w[j] -= cfg.lr * (g * xj + cfg.l2 * model.w[j]); // SGD + L2
      }
      model.b -= cfg.lr * g; // bias update
    }

    // epoch-end validation + early stop
    float v = accuracy_dataset(model, valid, cfg.thr); // epoch valid acc
    if (v > best_valid) { best_valid = v; bad = 0; } else { bad++; } // track improvement
    if (cfg.patience > 0 && bad >= cfg.patience) break; // early stop
  }

  out_train_acc = accuracy_dataset(model, train, cfg.thr); // train accuracy
  out_valid_acc = accuracy_dataset(model, valid, cfg.thr); // valid accuracy
  return model; // trained model
}
