from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SUBDIR = ROOT / "data" / "processed" / "submissions"
SUBDIR.mkdir(parents=True, exist_ok=True)

train_path = RAW / "train.tsv"
test_path  = RAW / "test.tsv"
sample_path = RAW / "sample-submission.csv"  # NOTE: hyphen in your repo

def die(msg: str) -> None:
    raise SystemExit(f"\nERROR: {msg}\n")

for p in (train_path, test_path, sample_path):
    if not p.exists():
        die(f"Missing file: {p}")

def read_tsv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False)

def normalize_text(s: pd.Series) -> pd.Series:
    s = s.astype(str).fillna("")
    return s.str.replace(r"\s+", " ", regex=True).str.strip()

def pick_id_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if c.lower() in ("id", "row_id", "index"):
            return c
    return df.columns[0]

def infer_label_col(train: pd.DataFrame, sample_pred_col: str) -> str:
    if sample_pred_col in train.columns:
        return sample_pred_col
    for c in train.columns:
        if c.lower() in ("label", "target", "sentiment", "class", "y"):
            return c
    die(f"Could not infer label column. Train columns: {list(train.columns)}")

def pick_text_col(train: pd.DataFrame, id_col: str, label_col: str) -> str:
    for c in train.columns:
        if c.lower() in ("text", "sentence", "review", "comment", "content", "tweet"):
            return c
    for c in train.columns:
        if c not in (id_col, label_col):
            return c
    die("Could not infer text column.")

train = read_tsv(train_path)
test  = read_tsv(test_path)
sample = pd.read_csv(sample_path)

if sample.shape[1] < 2:
    die("sample-submission.csv must have >= 2 columns (id + prediction).")

sub_id_col = sample.columns[0]
sub_pred_col = sample.columns[1]

train_id_col = sub_id_col if sub_id_col in train.columns else pick_id_col(train)
test_id_col  = sub_id_col if sub_id_col in test.columns  else pick_id_col(test)

label_col = infer_label_col(train, sub_pred_col)
text_col  = pick_text_col(train, train_id_col, label_col)

test_text_col = text_col if text_col in test.columns else None
if test_text_col is None:
    # try common names, else first non-id
    for c in test.columns:
        if c.lower() in ("text", "sentence", "review", "comment", "content", "tweet"):
            test_text_col = c
            break
if test_text_col is None:
    test_text_col = next(c for c in test.columns if c != test_id_col)

train[text_col] = normalize_text(train[text_col])
test[test_text_col] = normalize_text(test[test_text_col])

X = train[text_col].astype(str)
y = train[label_col].astype(str)

stratify = y if y.nunique() > 1 else None
X_tr, X_va, y_tr, y_va = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=stratify
)

tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.95, sublinear_tf=True)
to_dense = FunctionTransformer(lambda x: x.toarray(), accept_sparse=True)

models = {
    "gradient_boosting": Pipeline([("tfidf", tfidf), ("dense", to_dense),
                                  ("clf", GradientBoostingClassifier(random_state=42))]),
    "naive_bayes": Pipeline([("tfidf", tfidf),
                             ("clf", MultinomialNB())]),
    "random_forest": Pipeline([("tfidf", tfidf), ("dense", to_dense),
                               ("clf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1))]),
    "svm": Pipeline([("tfidf", tfidf),
                     ("clf", LinearSVC())]),
    "logistic_regression": Pipeline([("tfidf", tfidf),
                                     ("clf", LogisticRegression(max_iter=2000, n_jobs=-1))]),
}

print("\n=== SCHEMA DETECTED ===")
print(f"train_id_col={train_id_col}  test_id_col={test_id_col}")
print(f"text_col={text_col}  test_text_col={test_text_col}")
print(f"label_col={label_col}")
print("=======================\n")

scores = []
test_text = test[test_text_col].astype(str)

for name, pipe in models.items():
    print(f"=== TRAINING: {name} ===")
    pipe.fit(X_tr, y_tr)

    pred_va = pipe.predict(X_va)
    acc = accuracy_score(y_va, pred_va)
    scores.append((name, float(acc)))
    print(f"Accuracy (val): {acc:.6f}")

    pred_test = pipe.predict(test_text)

    sub = pd.DataFrame({sub_id_col: test[test_id_col].values, sub_pred_col: pred_test})
    for extra in sample.columns[2:]:
        sub[extra] = test[extra].values if extra in test.columns else sample[extra].iloc[0]

    out = SUBDIR / f"{name}_submission_1.csv"
    sub.to_csv(out, index=False)
    print(f"Saved: {out}\n")

print("=== MODEL COMPARISON (ACCURACY) ===")
for i, (name, acc) in enumerate(sorted(scores, key=lambda x: x[1], reverse=True), 1):
    print(f"{i:>2}. {name:<20} {acc:.6f}")

print(f"\nSubmissions written to: {SUBDIR}")
