import pytest

class TestSupervisedModel:
    def test_import(self):
        try:
            from sa.supervised import models
        except ImportError:
            pytest.skip("sa.supervised.models not yet implemented")

    def test_fit_predict_shape(self, sample_texts, sample_labels):
        try:
            from sa.supervised.models.logreg import LogRegModel
        except ImportError:
            pytest.skip("LogRegModel not yet implemented")
        model = LogRegModel()
        model.fit(sample_texts, sample_labels)
        assert len(model.predict(sample_texts)) == len(sample_texts)

    def test_predict_binary_outputs(self, sample_texts, sample_labels):
        try:
            from sa.supervised.models.logreg import LogRegModel
        except ImportError:
            pytest.skip("LogRegModel not yet implemented")
        model = LogRegModel()
        model.fit(sample_texts, sample_labels)
        assert all(p in (0, 1) for p in model.predict(sample_texts))

    def test_predict_proba_in_range(self, sample_texts, sample_labels):
        try:
            from sa.supervised.models.logreg import LogRegModel
        except ImportError:
            pytest.skip("LogRegModel not yet implemented")
        model = LogRegModel()
        model.fit(sample_texts, sample_labels)
        assert all(0.0 <= p <= 1.0 for p in model.predict_proba(sample_texts))
