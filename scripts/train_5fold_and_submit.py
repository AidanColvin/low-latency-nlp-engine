from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "submissions_5fold"
OUT.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(RAW/"train.tsv", sep="\t", dtype=str, keep_default_na=False)
test  = pd.read_csv(RAW/"test.tsv",  sep="\t", dtype=str, keep_default_na=False)
sample = pd.read_csv(RAW/"sample-submission.csv")

# Your detected schema (from your run)
text_col, label_col = "review", "label"
test_id_col = "id"

sub_id_col, sub_pred_col = sample.columns[0], sample.columns[1]

def norm(s):
    return s.astype(str).fillna("").str.replace(r"\s+"," ",regex=True).str.strip()

train[text_col] = norm(train[text_col])
test[text_col]  = norm(test[text_col])

X = train[text_col]
y = train[label_col]

# TF-IDF kept consistent across models
tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=50000, min_df=2)

# Models (same as your working run; sparse-friendly)
models = {
    "naive_bayes": Pipeline([("tfidf", tfidf), ("clf", MultinomialNB())]),
    "logistic_regression": Pipeline([("tfidf", tfidf), ("clf", LogisticRegression(max_iter=1500))]),
    "svm": Pipeline([("tfidf", tfidf), ("clf", LinearSVC())]),
    # keep the names you want; sparse-friendly approximations for text
    "random_forest": Pipeline([("tfidf", tfidf), ("clf", SGDClassifier(loss="log_loss", max_iter=1500, random_state=42))]),
    "gradient_boosting": Pipeline([("tfidf", tfidf), ("clf", SGDClassifier(loss="hinge",   max_iter=1500, random_state=42))]),
}

print("\n=== 5-FOLD CV (Stratified) ===")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rows = []
for name, pipe in models.items():
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    rows.append((name, float(scores.mean()), float(scores.std())))
    print(f"{name:<20} mean={scores.mean():.6f}  std={scores.std():.6f}  folds={np.round(scores, 6)}")

rows_sorted = sorted(rows, key=lambda r: r[1], reverse=True)

print("\n=== RANKED (5-FOLD MEAN ACCURACY) ===")
for i, (name, mean, std) in enumerate(rows_sorted, 1):
    print(f"{i:>2}. {name:<20} {mean:.6f} ± {std:.6f}")

print("\n=== REFIT ON ALL TRAIN + WRITE SUBMISSIONS ===")
for name, pipe in models.items():
    print(f"\nRefit: {name}", flush=True)
    pipe.fit(X, y)

    pred_test = pipe.predict(test[text_col])
    sub = pd.DataFrame({sub_id_col: test[test_id_col].values, sub_pred_col: pred_test})

    # preserve extra columns if they exist in sample (rare)
    for extra in sample.columns[2:]:
        sub[extra] = test[extra].values if extra in test.columns else sample[extra].iloc[0]

    out = OUT / f"{name}_submission_5foldfit_1.csv"
    sub.to_csv(out, index=False)
    print(f"Saved: {out}", flush=True)

print(f"\nDone. Submissions: {OUT}")
