import os, pytest

REQUIRED_DIRS = [
    "src/cpp/src", "src/cpp/include", "src/cpp/build",
    "src/models/fasttext/fasttext_repo",
    "src/models/lgbm/lgbm_repo",
    "src/models/tinybert/tinybert_repo",
    "src/scripts",
    "src/outputs/04_train_cpp",
    "src/outputs/05_predict_cpp",
    "data/raw", "tests",
]
REQUIRED_FILES = [
    "src/scripts/dl_pipeline.py",
    "src/cpp/CMakeLists.txt",
    "data/raw/train.tsv",
    "data/raw/test.tsv",
    "requirements.txt",
    "pytest.ini",
]

class TestRepoLayout:
    @pytest.mark.parametrize("d", REQUIRED_DIRS)
    def test_dir_exists(self, d):
        assert os.path.isdir(d), f"Missing dir: {d}"

    @pytest.mark.parametrize("f", REQUIRED_FILES)
    def test_file_exists(self, f):
        assert os.path.isfile(f), f"Missing file: {f}"

class TestRequirements:
    def test_not_empty(self):
        with open("requirements.txt") as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        assert len(lines) > 0

class TestGitignore:
    def test_gitignore_exists(self):
        assert os.path.isfile(".gitignore"), ".gitignore missing — run: touch .gitignore"
