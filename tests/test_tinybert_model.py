import os, ast, pytest

REPO = "src/models/tinybert/tinybert_repo"
FILES = ["tb_config.py","tb_export.py","tb_main.py"]

class TestTinyBERTFiles:
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
    def test_tb_config_has_model_params(self):
        with open(f"{REPO}/tb_config.py") as f: src = f.read()
        assert any(k in src.lower() for k in ("model","bert","batch","epoch","lr")),             "tb_config.py looks empty"
    def test_tb_export_has_export(self):
        with open(f"{REPO}/tb_export.py") as f: src = f.read()
        assert any(k in src.lower() for k in ("export","save","torch","model")),             "tb_export.py has no export/save logic"
    def test_cpp_files_exist(self):
        for f in ["tb_main.cpp","tb_loader.cpp"]:
            assert os.path.isfile(os.path.join(REPO, f)), f"{f} missing"

class TestTransformersInstall:
    def test_torch_importable(self):
        try:
            import torch
            assert torch.__version__
        except ImportError: pytest.skip("torch not installed — pip install torch")
    def test_transformers_importable(self):
        try:
            import transformers
            assert transformers.__version__
        except ImportError: pytest.skip("transformers not installed — pip install transformers")
