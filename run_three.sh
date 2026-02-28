#!/usr/bin/env bash
set -euo pipefail

export RUN_ID="$(date +%Y%m%d_%H%M%S)"
export OUT_ROOT="runs/${RUN_ID}"
export PY_OUT="${OUT_ROOT}/python"
export CPP_OUT="${OUT_ROOT}/cpp"
export LOGDIR="${OUT_ROOT}/logs"

mkdir -p "${PY_OUT}" "${CPP_OUT}" "${LOGDIR}"

echo "=== RUN_ID: ${RUN_ID} ==="
echo "Outputs: ${OUT_ROOT}"
echo ""

python3 -m pip install -q -r requirements.txt || true
python3 -m pip install -q -r requirements-dev.txt || true

# --- Model 1: Python SVM (your best config) ---
python3 - <<'PY' | tee "${LOGDIR}/01_svm.log"
import os, json, joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

ROOT = Path(".").resolve()
train = pd.read_csv(ROOT/"data/raw/train.tsv", sep="\t")
test  = pd.read_csv(ROOT/"data/raw/test.tsv",  sep="\t")
sample= pd.read_csv(ROOT/"data/raw/sample-submission.csv")

text_col  = "review" if "review" in train.columns else train.columns[0]
label_col = "label"  if "label"  in train.columns else train.columns[-1]
test_id_col = "id"   if "id" in test.columns else test.columns[0]
sub_id_col, sub_pred_col = sample.columns[0], sample.columns[1]

X = train[text_col].astype(str)
y = train[label_col].astype(int)

vec = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.95, strip_accents="unicode",
                      lowercase=True, sublinear_tf=True)
clf = SGDClassifier(loss="hinge", alpha=1e-4, max_iter=4000, tol=1e-3, random_state=42)
pipe = Pipeline([("tfidf", vec), ("clf", clf)])

print("=== MODEL #1: SVM (SGD hinge) TFIDF word(1,2) ===")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
mu, sd = float(scores.mean()), float(scores.std())
print(f"ASSUMED ACCURACY (5-fold): {mu:.6f} ± {sd:.6f}")

print("Training on FULL train ...")
pipe.fit(X, y)

pred = pipe.predict(test[text_col].astype(str))
out_dir = Path(os.environ["PY_OUT"])/"svm_word12"
out_dir.mkdir(parents=True, exist_ok=True)

sub = pd.DataFrame({sub_id_col: test[test_id_col].values, sub_pred_col: pred})
sub_path = out_dir/"submission.csv"
sub.to_csv(sub_path, index=False)

joblib.dump(pipe, out_dir/"model.joblib")
(out_dir/"metrics.json").write_text(json.dumps({"cv_mean_acc": mu, "cv_std_acc": sd}, indent=2))

print("Saved:", sub_path)
PY

# --- Model 2: Python LR (strong config) ---
python3 - <<'PY' | tee "${LOGDIR}/02_lr.log"
import os, json, joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(".").resolve()
train = pd.read_csv(ROOT/"data/raw/train.tsv", sep="\t")
test  = pd.read_csv(ROOT/"data/raw/test.tsv",  sep="\t")
sample= pd.read_csv(ROOT/"data/raw/sample-submission.csv")

text_col  = "review" if "review" in train.columns else train.columns[0]
label_col = "label"  if "label"  in train.columns else train.columns[-1]
test_id_col = "id"   if "id" in test.columns else test.columns[0]
sub_id_col, sub_pred_col = sample.columns[0], sample.columns[1]

X = train[text_col].astype(str)
y = train[label_col].astype(int)

vec = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.95, strip_accents="unicode",
                      lowercase=True, sublinear_tf=True)
clf = LogisticRegression(C=4.0, max_iter=6000)
pipe = Pipeline([("tfidf", vec), ("clf", clf)])

print("=== MODEL #2: LR TFIDF word(1,2) C=4.0 ===")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
mu, sd = float(scores.mean()), float(scores.std())
print(f"ASSUMED ACCURACY (5-fold): {mu:.6f} ± {sd:.6f}")

print("Training on FULL train ...")
pipe.fit(X, y)

pred = pipe.predict(test[text_col].astype(str))
out_dir = Path(os.environ["PY_OUT"])/"lr_word12_c4"
out_dir.mkdir(parents=True, exist_ok=True)

sub = pd.DataFrame({sub_id_col: test[test_id_col].values, sub_pred_col: pred})
sub_path = out_dir/"submission.csv"
sub.to_csv(sub_path, index=False)

joblib.dump(pipe, out_dir/"model.joblib")
(out_dir/"metrics.json").write_text(json.dumps({"cv_mean_acc": mu, "cv_std_acc": sd}, indent=2))

print("Saved:", sub_path)
PY

# --- Model 3: C++ stages ---
python3 -u src/src/stages/stage_04_train_cpp.py 2>&1 | tee "${LOGDIR}/03_cpp_train.log"
python3 -u src/src/stages/stage_05_predict_cpp.py 2>&1 | tee "${LOGDIR}/04_cpp_predict.log"

if [ -d "src/outputs/04_train_cpp" ]; then rsync -a "src/outputs/04_train_cpp/" "${CPP_OUT}/04_train_cpp/"; fi
if [ -d "src/outputs/05_predict_cpp" ]; then rsync -a "src/outputs/05_predict_cpp/" "${CPP_OUT}/05_predict_cpp/"; fi

echo ""
echo "=== DONE: ${OUT_ROOT} ==="
echo "Python files:"; find "${PY_OUT}" -type f | sed -n '1,200p'
echo ""
echo "C++ files:"; find "${CPP_OUT}" -type f | sed -n '1,200p'
