#include "sgd_trainer.h" // decl
#include "metrics.h" // accuracy
#include <random> // shuffle
#include <numeric> // iota
#include <algorithm> // shuffle

LogRegModel train_sgd_logreg(const Dataset& train, const Dataset& valid, int dim, const TrainConfig& cfg,
                            float& out_train_acc, float& out_valid_acc) {
  LogRegModel model = make_model(dim); // init model

  std::vector<int> order(train.size()); // indices
  std::iota(order.begin(), order.end(), 0); // 0..n-1

  std::mt19937 rng(123); // deterministic shuffle

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
  }

  out_train_acc = accuracy_dataset(model, train, cfg.thr); // train accuracy
  out_valid_acc = accuracy_dataset(model, valid, cfg.thr); // valid accuracy
  return model; // trained model
}

LogRegModel SGDTrainer::fit_all(const Dataset& train) { return fit(train, train); }
