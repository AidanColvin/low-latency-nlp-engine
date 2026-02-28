import os, ast, pytest

REPO = "src/models/lgbm/lgbm_repo"
FILES = ["lgb_config.py","lgb_main.py","lgb_train.py"]

class TestLGBMFiles:
    @pytest.mark.parametrize("f", FILES)
    def test_file_exists(self, f):
        assert os.path.isfile(os.path.join(REPO, f)), f"{f} missing"
    @pytest.mark.parametrize("f", FILES)
    def test_not_empty(self, f):
        assert os.path.getsize(os.path.join(REPO, f)) > 0
    @pytest.mark.parametrize("f", FILES)
    def test_valid_syntax(self, f):
        with open(os.path.join(REPO, f)) as fp: src = fp.read()
        try: ast.parse(src)
        except SyntaxError as e: pytest.fail(f"{f} syntax error: {e}")
    def test_lgb_config_has_params(self):
        with open(f"{REPO}/lgb_config.py") as f: src = f.read()
        assert "=" in src and len(src.strip()) > 20, "lgb_config.py appears to be empty"
    def test_cpp_files_exist(self):
        for f in ["lgb_main.cpp","lgb_loader.cpp"]:
            assert os.path.isfile(os.path.join(REPO, f)), f"{f} missing"

class TestLGBMInstall:
    def test_importable(self):
        try: import lightgbm
        except OSError as e: pytest.fail(f"lgbm lib missing: {e}")
        except ImportError: pytest.skip("lightgbm not installed")
    def test_version(self):
        try:
            import lightgbm as lgb
            assert lgb.__version__
        except (ImportError, OSError): pytest.skip("not available")
    def test_fit_predict(self, sample_texts, sample_labels):
        try:
            import lightgbm as lgb
            from sklearn.feature_extraction.text import TfidfVectorizer
        except (ImportError, OSError): pytest.skip("lgbm/sklearn not available")
        X = TfidfVectorizer(max_features=200).fit_transform(sample_texts)
        clf = lgb.LGBMClassifier(n_estimators=5, verbose=-1)
        clf.fit(X, sample_labels)
        preds = clf.predict(X)
        assert set(preds).issubset({0,1})
        probas = clf.predict_proba(X)[:,1]
        assert all(0.0 <= p <= 1.0 for p in probas)
