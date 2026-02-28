import pytest

class TestMetrics:
    def test_accuracy_perfect(self):
        try:
            from sa.evaluation.metrics import accuracy
        except ImportError:
            pytest.skip("accuracy not yet implemented")
        assert accuracy([1, 0, 1], [1, 0, 1]) == 1.0

    def test_accuracy_zero(self):
        try:
            from sa.evaluation.metrics import accuracy
        except ImportError:
            pytest.skip("accuracy not yet implemented")
        assert accuracy([1, 1, 1], [0, 0, 0]) == 0.0

    def test_f1_perfect(self):
        try:
            from sa.evaluation.metrics import f1
        except ImportError:
            pytest.skip("f1 not yet implemented")
        assert f1([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)

    def test_classification_report_keys(self):
        try:
            from sa.evaluation.metrics import classification_report
        except ImportError:
            pytest.skip("classification_report not yet implemented")
        report = classification_report([1, 0, 1], [1, 0, 0])
        assert "accuracy" in report and "f1" in report
