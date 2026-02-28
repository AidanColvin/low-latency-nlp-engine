import os, ast, pytest

DL_PIPELINE = "src/scripts/dl_pipeline.py"

class TestDLPipeline:
    def test_exists(self):
        assert os.path.isfile(DL_PIPELINE)
    def test_not_empty(self):
        assert os.path.getsize(DL_PIPELINE) > 0
    def test_valid_syntax(self):
        with open(DL_PIPELINE) as f: src = f.read()
        try: ast.parse(src)
        except SyntaxError as e: pytest.fail(f"Syntax error: {e}")
    def test_has_imports(self):
        with open(DL_PIPELINE) as f: src = f.read()
        tree = ast.parse(src)
        imports = [n for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))]
        assert len(imports) > 0, "dl_pipeline.py has no imports"
    def test_has_meaningful_code(self):
        with open(DL_PIPELINE) as f: src = f.read()
        lines = [l for l in src.splitlines() if l.strip() and not l.strip().startswith("#")]
        assert len(lines) >= 5
