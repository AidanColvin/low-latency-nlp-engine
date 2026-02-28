import os, pytest

class TestFastTextDirectory:
    def test_exists(self):
        assert os.path.isdir("src/models/fasttext"), "src/models/fasttext not found"

    def test_not_empty(self):
        assert len(os.listdir("src/models/fasttext")) > 0

class TestFastTextWrapper:
    def test_train_and_predict(self, tmp_path, sample_pairs):
        try:
            import fasttext
        except ImportError:
            pytest.skip("fasttext not installed")
        train_file = tmp_path / "train.txt"
        train_file.write_text("\n".join(f"__label__{l} {t}" for t, l in sample_pairs))
        model = fasttext.train_supervised(str(train_file), epoch=5, lr=0.5)
        label, prob = model.predict("this is great")
        assert label[0] in ("__label__0", "__label__1")
        assert 0.0 <= prob[0] <= 1.0
