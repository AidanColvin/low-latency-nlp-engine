import pytest

class TestHashingVectoriser:
    def test_import(self):
        try:
            from sa.features import vectoriser
        except ImportError:
            pytest.skip("sa.features.vectoriser not yet implemented")

    def test_transform_returns_same_length(self, sample_texts):
        try:
            from sa.features.vectoriser import HashingVectoriser
        except ImportError:
            pytest.skip("HashingVectoriser not yet implemented")
        v = HashingVectoriser(dim=1 << 16, n_min=1, n_max=2)
        assert len(v.transform(sample_texts)) == len(sample_texts)

    def test_empty_string_does_not_crash(self):
        try:
            from sa.features.vectoriser import HashingVectoriser
        except ImportError:
            pytest.skip("HashingVectoriser not yet implemented")
        v = HashingVectoriser(dim=1 << 16, n_min=1, n_max=2)
        assert len(v.transform([""])) == 1
