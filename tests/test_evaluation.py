import pytest

class TestSklearnMetrics:
    def test_sklearn_importable(self):
        try: from sklearn import metrics
        except ImportError: pytest.fail("scikit-learn not installed")
    def test_accuracy(self):
        from sklearn.metrics import accuracy_score
        assert accuracy_score([1,0,1],[1,0,1]) == 1.0
        assert accuracy_score([1,1],[0,0]) == 0.0
    def test_f1(self):
        from sklearn.metrics import f1_score
        assert abs(f1_score([1,0,1,0],[1,0,1,0]) - 1.0) < 1e-6
    def test_roc_auc(self):
        from sklearn.metrics import roc_auc_score
        s = roc_auc_score([0,0,1,1],[0.1,0.2,0.8,0.9])
        assert 0.0 <= s <= 1.0
    def test_classification_report(self):
        from sklearn.metrics import classification_report
        r = classification_report([1,0,1],[1,0,0], output_dict=True)
        assert "accuracy" in r
