# Low Latency NLP Engine

> A hybrid Python–C++ machine learning system for large-scale sentiment classification — featuring a custom C++ training and inference backend, automated model selection via cross-validation, and a reproducible end-to-end pipeline from raw text to deployment-ready artifacts.

Built for [SP26 INLS 642: What Makes a Positive Review?](https://kaggle.com/competitions/sp-26-inls-642-what-makes-a-positive-review) · Data: [Amazon Customer Reviews](https://registry.opendata.aws/amazon-reviews/)

---

## Why This Project

Most NLP projects stop at a Jupyter notebook. This one doesn't.

The goal was to build a system — not just a model. That means reproducible training runs, a real C++ inference backend, modular stages you can swap out, and artifacts you can actually deploy. The dataset is a Kaggle competition on Amazon review sentiment; the engineering is the point.

---

## Architecture

```
Raw Text (TSV)
     │
     ▼
Text Processing ── TF-IDF / Hashing Vectorizer / n-gram extraction
     │
     ├──────────────────────────┬──────────────────────────┐
     ▼                          ▼                          ▼
Python ML Layer          Deep Learning Layer         C++ Engine
─────────────────        ───────────────────         ──────────────────
Logistic Regression      TinyBERT (extensible)       Hash n-gram features
SVM / SGD                FastText module             SGD optimizer
Naive Bayes              LightGBM module             Binary serialization
Random Forest                                        Sub-millisecond inference
     │                          │                          │
     └──────────────────────────┴──────────────────────────┘
                                │
                                ▼
                        Evaluation Layer
                  5-fold CV · accuracy · model ranking
                                │
                                ▼
                        Full Dataset Retraining
                                │
                                ▼
                    Submission + Model Artifacts
              (CSV · Parquet · .joblib · .bin · metrics.json)
```

**Three execution layers, one unified pipeline.** Python handles rapid experimentation. C++ handles performance. The evaluation layer automatically ranks models and selects the best configuration before final training.

---

## System Components

### Python ML Layer (`scripts/`)

Classical supervised learning with automated hyperparameter search and cross-validation:

- **Models:** Logistic Regression (L2), LinearSVC, SGD Classifier, Naive Bayes, Random Forest, Gradient Boosting
- **Features:** TF-IDF with word n-grams (1–3) and character n-grams (2–5), sublinear TF scaling, sparse matrix representation
- **Selection:** 5-fold stratified CV across model/hyperparameter combinations; best config auto-selected for full training

### C++ High-Performance Engine (`src/cpp/`)

Custom-built training and inference backend — not a wrapper around an existing library:

- **Vectorizer:** Hash-based n-gram feature generation (fixed-width, no vocabulary bottleneck)
- **Optimizer:** Online SGD with configurable learning rate and regularization
- **Serialization:** Binary model format (`.bin`) for fast load and deployment
- **Inference:** Designed for sub-millisecond latency at scale

This is the differentiator. Most ML repos never touch C++.

### Modular Model Layer (`src/models/`)

Extensible architecture for swapping in more powerful models:

```
src/models/
├── fasttext/       # embedding-based classification
├── lgbm/           # gradient boosting on sparse features
└── tinybert/       # transformer fine-tuning entrypoint
```

Adding a new model means implementing one interface, not rewriting the pipeline.

### Pipeline Orchestration

Three ways to run the system, matching different use cases:

| Entry Point | Purpose |
|---|---|
| `run_three.sh` | Full end-to-end (Python + C++ stages) |
| `scripts/train_classical.py` | Python models only |
| `dl_pipeline.py` | Deep learning path |

Stages are discrete and independently runnable — closer to an Airflow DAG than a monolithic script.

### Experiment Tracking (`runs/`)

Every training run produces a timestamped directory:

```
runs/<timestamp>/
├── python/
│   └── <model>/
│       ├── submission.csv
│       ├── model.joblib
│       └── metrics.json
├── cpp/
│   ├── 04_train_cpp/
│   └── 05_predict_cpp/
└── logs/
```

Reproducibility isn't an afterthought — it's structural.

---

## Model Results

| Model | CV Accuracy | Notes |
|---|---|---|
| Logistic Regression (C=4.0) | ~0.904 | word n-grams (1–2), TF-IDF |
| SVM / SGD Hinge | ~0.903 | linear kernel |
| LR + SVC Ensemble (word + char n-grams) | **~0.935** | best leaderboard score |
| C++ Hash-Ngram SGD | Competitive | fastest inference |

The ensemble result combines TF-IDF word n-grams (1–3) and character n-grams (2–5) across calibrated LR and SVC models. Character n-grams capture subword patterns and spelling variation that word models miss.

---

## Project Structure

```
.
├── data/
│   ├── raw/                        # source TSV files
│   └── processed/
│       ├── submissions/
│       ├── submissions_5fold/
│       └── submissions_final/
├── scripts/
│   ├── train_classical.py          # Python ML training
│   └── train_5fold_and_submit.py   # CV + submission generation
├── src/
│   ├── cpp/
│   │   ├── app/                    # C++ entrypoints
│   │   ├── include/                # headers
│   │   └── src/                    # implementations
│   ├── models/
│   │   ├── fasttext/
│   │   ├── lgbm/
│   │   └── tinybert/
│   └── src/stages/
│       ├── stage_04_train_cpp.py
│       └── stage_05_predict_cpp.py
├── runs/                           # experiment artifacts
├── tests/
│   ├── test_supervised.py
│   ├── test_deep_learning.py
│   ├── test_cpp.py
│   └── test_pipeline*.py
├── dl_pipeline.py
└── run_three.sh
```

---

## Running the Pipeline

**Full pipeline (Python + C++):**
```bash
bash run_three.sh
```

**Python models only:**
```bash
python3 scripts/train_classical.py
```

**Individual C++ stages:**
```bash
python3 src/src/stages/stage_04_train_cpp.py   # train
python3 src/src/stages/stage_05_predict_cpp.py  # predict
```

**Tests:**
```bash
pytest tests/
```

---

## Design Principles

**Reproducibility.** Deterministic pipelines, saved artifacts, timestamped runs. You can re-run any experiment and get the same result.

**Separation of concerns.** Data, features, models, evaluation, and output are discrete stages — not entangled in a single script.

**Performance where it matters.** Python for flexibility during experimentation. C++ for inference where latency is the constraint.

**Extensibility.** The model layer is designed for drop-in replacements. Swap TinyBERT for DistilBERT, or FastText for a custom embedding — the pipeline doesn't change.

---

## Roadmap

- [ ] Transformer fine-tuning (DistilBERT / TinyBERT) via `src/models/tinybert/`
- [ ] Stacked ensemble: OOF meta-features from LR + SVC + LGBM
- [ ] GPU acceleration for deep learning stages
- [ ] REST inference endpoint wrapping the C++ binary

---

## Acknowledgments

Competition: Ray Wang. *SP26 INLS 642: What Makes a Positive Review?* Kaggle, 2026.

Data: [Amazon Customer Reviews Dataset](https://registry.opendata.aws/amazon-reviews/)

---

## License

MIT
