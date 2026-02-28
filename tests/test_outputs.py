import os, json, pytest

TRAIN_OUT = "src/outputs/04_train_cpp/logreg_hashngram_5fold_v1_final"
PREDICT_OUT = "src/outputs/05_predict_cpp"

class TestTrainOutputs:
    def test_output_dir_exists(self):
        assert os.path.isdir(TRAIN_OUT), f"{TRAIN_OUT} missing"
    def test_model_bin_exists(self):
        assert os.path.isfile(f"{TRAIN_OUT}/model.bin"), "model.bin missing"
    def test_model_bin_not_empty(self):
        assert os.path.getsize(f"{TRAIN_OUT}/model.bin") > 0, "model.bin is 0 bytes"
    def test_metrics_json_exists(self):
        assert os.path.isfile(f"{TRAIN_OUT}/metrics.json")
    def test_metrics_json_valid(self):
        with open(f"{TRAIN_OUT}/metrics.json") as f:
            data = json.load(f)
        assert isinstance(data, dict), "metrics.json is not a JSON object"
    def test_metrics_has_accuracy(self):
        with open(f"{TRAIN_OUT}/metrics.json") as f:
            data = json.load(f)
        keys = " ".join(data.keys()).lower()
        assert any(k in keys for k in ("acc","f1","auc","score","metric"))
    def test_checkpoint_json_exists(self):
        assert os.path.isfile(f"{TRAIN_OUT}/checkpoint.json")
    def test_checkpoint_valid(self):
        with open(f"{TRAIN_OUT}/checkpoint.json") as f:
            data = json.load(f)
        assert isinstance(data, dict)

class TestPredictOutputs:
    def test_predict_dir_exists(self):
        assert os.path.isdir(PREDICT_OUT)
    def test_submission_csv_exists(self):
        csv = f"{PREDICT_OUT}/logreg_hashngram_5fold_v1_submission.csv"
        assert os.path.isfile(csv), "submission CSV missing"
    def test_submission_csv_not_empty(self):
        csv = f"{PREDICT_OUT}/logreg_hashngram_5fold_v1_submission.csv"
        assert os.path.getsize(csv) > 0
    def test_submission_csv_format(self):
        import csv as csvmod
        path = f"{PREDICT_OUT}/logreg_hashngram_5fold_v1_submission.csv"
        with open(path, newline="") as f:
            rows = list(csvmod.reader(f))
        assert len(rows) >= 2, "submission CSV has fewer than 2 rows"
        assert len(rows[0]) >= 2, "submission CSV needs at least 2 columns (id, label)"
    def test_parquet_exists(self):
        try:
            import pandas
        except ImportError:
            pytest.skip("pandas not installed")
        parq = f"{PREDICT_OUT}/logreg_hashngram_5fold_v1_submission.parquet"
        assert os.path.isfile(parq), "submission parquet missing"
    def test_parquet_readable(self):
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not installed")
        parq = f"{PREDICT_OUT}/logreg_hashngram_5fold_v1_submission.parquet"
        if not os.path.isfile(parq): pytest.skip("parquet not generated yet")
        df = pd.read_parquet(parq)
        assert len(df) > 0
