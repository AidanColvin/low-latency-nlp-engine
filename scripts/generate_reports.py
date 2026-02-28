import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(".")
TRAIN = ROOT / "data/raw/train.tsv"

OUT_T = ROOT / "data/processed/tables"
OUT_V = ROOT / "data/processed/visualizations"

OUT_T.mkdir(parents=True, exist_ok=True)
OUT_V.mkdir(parents=True, exist_ok=True)

print("Loading data...")
df = pd.read_csv(TRAIN, sep="\t")
X = df["review"].astype(str).values
y = df["label"].values

tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.95)

models = {
    "svm": Pipeline([("tfidf", tfidf), ("clf", LinearSVC(C=0.5))]),
    "log_reg": Pipeline([("tfidf", tfidf), ("clf", LogisticRegression(max_iter=4000))]),
    "sgd": Pipeline([("tfidf", tfidf), ("clf", SGDClassifier(loss="hinge"))]),
    "naive_bayes": Pipeline([("tfidf", tfidf), ("clf", MultinomialNB())]),
    "random_forest": Pipeline([("tfidf", tfidf), ("clf", RandomForestClassifier(n_estimators=200))]),
}

results = []

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"Training {name}...")
    scores = []
    all_true, all_pred = [], []

    for tr, va in skf.split(X, y):
        model.fit(X[tr], y[tr])
        pred = model.predict(X[va])
        acc = accuracy_score(y[va], pred)
        scores.append(acc)
        all_true.extend(y[va])
        all_pred.extend(pred)

    mean = np.mean(scores)
    std = np.std(scores)

    results.append((name, mean, std))

    # Fold plot
    plt.figure()
    plt.plot(scores, marker='o')
    plt.title(name)
    plt.savefig(OUT_V / f"{name}_folds.png")
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(all_true, all_pred)
    plt.figure()
    plt.imshow(cm)
    plt.title(name)
    plt.savefig(OUT_V / f"{name}_cm.png")
    plt.close()

# Save table
df_out = pd.DataFrame(results, columns=["model","mean_acc","std"])
df_out.to_csv(OUT_T / "model_summary.csv", index=False)

# Overall plot
df_out = df_out.sort_values("mean_acc", ascending=False)
plt.figure()
plt.barh(df_out["model"], df_out["mean_acc"])
plt.gca().invert_yaxis()
plt.title("Model Comparison")
plt.savefig(OUT_V / "overall.png")
plt.close()

# REPORT
with open("REPORT.md", "w") as f:
    f.write("# Results\n\n")
    f.write(df_out.to_string(index=False))

print("DONE")
