import pytest

class TestDeepLearningModel:
    def test_import(self):
        try:
            from sa import deep_learning
        except ImportError:
            pytest.skip("sa.deep_learning not yet implemented")

    def test_tinybert_loads(self):
        try:
            from sa.deep_learning.models.tinybert import TinyBertClassifier
        except ImportError:
            pytest.skip("TinyBertClassifier not yet implemented")
        assert TinyBertClassifier() is not None

    def test_tinybert_predict_shape(self, sample_texts):
        try:
            from sa.deep_learning.models.tinybert import TinyBertClassifier
        except ImportError:
            pytest.skip("TinyBertClassifier not yet implemented")
        assert len(TinyBertClassifier().predict(sample_texts)) == len(sample_texts)

class TestDataModule:
    def test_dataset_length(self, sample_pairs):
        try:
            from sa.deep_learning.data.dataset import SentimentDataset
        except ImportError:
            pytest.skip("SentimentDataset not yet implemented")
        texts, labels = zip(*sample_pairs)
        assert len(SentimentDataset(list(texts), list(labels))) == len(sample_pairs)
