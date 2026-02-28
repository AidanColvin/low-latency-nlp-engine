import os, pytest

class TestLGBMDirectory:
    def test_exists(self):
        assert os.path.isdir("src/models/lgbm"), "src/models/lgbm not found"

class TestLGBMClassifier:
    def test_fit_predict(self, sample_texts, sample_labels):
        try:
            import lightgbm as lgb
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            pytest.skip("lightgbm or sklearn not installed")
        vec = TfidfVectorizer(max_features=500)
        X = vec.fit_transform(sample_texts)
        clf = lgb.LGBMClassifier(n_estimators=10, verbose=-1)
        clf.fit(X, sample_labels)
        preds = clf.predict(X)
        assert len(preds) == len(sample_labels)
        assert set(preds).issubset({0, 1})
