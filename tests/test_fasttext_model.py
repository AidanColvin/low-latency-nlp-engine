import os, ast, pytest

REPO = "src/models/fasttext/fasttext_repo"
FILES = ["ft_config.py","ft_main.py","ft_train.py"]

class TestFastTextFiles:
    @pytest.mark.parametrize("f", FILES)
    def test_file_exists(self, f):
        assert os.path.isfile(os.path.join(REPO, f)), f"{f} missing from fasttext repo"
    @pytest.mark.parametrize("f", FILES)
    def test_not_empty(self, f):
        assert os.path.getsize(os.path.join(REPO, f)) > 0, f"{f} is 0 bytes"
    @pytest.mark.parametrize("f", FILES)
    def test_valid_syntax(self, f):
        with open(os.path.join(REPO, f)) as fp: src = fp.read()
        try: ast.parse(src)
        except SyntaxError as e: pytest.fail(f"{f} syntax error: {e}")
    def test_ft_config_has_config(self):
        with open(f"{REPO}/ft_config.py") as f: src = f.read()
        # config may just have file paths — check it has at least one assignment
        assert "=" in src and len(src.strip()) > 20, "ft_config.py appears to be empty"
    def test_ft_train_has_train(self):
        with open(f"{REPO}/ft_train.py") as f: src = f.read()
        assert "train" in src.lower(), "ft_train.py has no 'train' keyword"
    def test_cpp_files_exist(self):
        for f in ["ft_main.cpp","ft_predict.cpp"]:
            assert os.path.isfile(os.path.join(REPO, f)), f"{f} missing"

class TestFastTextInstall:
    def test_importable(self):
        try: import fasttext
        except ImportError: pytest.skip("fasttext not installed — pip install fasttext")
    def test_train_toy(self, tmp_path, sample_texts, sample_labels):
        try: import fasttext
        except ImportError: pytest.skip("fasttext not installed")
        f = tmp_path/"train.txt"
        f.write_text("\n".join(f"__label__{l} {t}" for t,l in zip(sample_texts,sample_labels)))
        model = fasttext.train_supervised(str(f), epoch=3, verbose=0)
        label, prob = model.predict("great product")
        assert label[0] in ("__label__0","__label__1")
        assert 0.0 <= prob[0] <= 1.0
