import os, ast, pytest

STAGE_04 = "src/src/stages/stage_04_train_cpp.py"
STAGE_05 = "src/src/stages/stage_05_predict_cpp.py"

class TestStageFiles:
    def test_stage_04_exists(self):
        assert os.path.isfile(STAGE_04), f"{STAGE_04} missing"
    def test_stage_05_exists(self):
        assert os.path.isfile(STAGE_05), f"{STAGE_05} missing"
    def test_stage_04_not_empty(self):
        assert os.path.getsize(STAGE_04) > 0
    def test_stage_05_not_empty(self):
        assert os.path.getsize(STAGE_05) > 0

class TestStageSyntax:
    @pytest.mark.parametrize("path", [STAGE_04, STAGE_05])
    def test_valid_syntax(self, path):
        with open(path) as f: src = f.read()
        try: ast.parse(src)
        except SyntaxError as e: pytest.fail(f"{os.path.basename(path)} syntax error: {e}")

    def test_stage_04_calls_train(self):
        with open(STAGE_04) as f: src = f.read()
        assert "train_cpp" in src or "subprocess" in src or "train" in src.lower(),             "stage_04 does not reference train_cpp or subprocess"

    def test_stage_05_calls_predict(self):
        with open(STAGE_05) as f: src = f.read()
        assert "predict_cpp" in src or "subprocess" in src or "predict" in src.lower(),             "stage_05 does not reference predict_cpp or subprocess"
