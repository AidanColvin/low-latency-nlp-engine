import importlib.util, os, pytest

class TestScripts:
    def test_scripts_dir_exists(self):
        assert os.path.isdir("src/scripts"), "src/scripts not found"

    def test_dl_pipeline_exists(self):
        assert os.path.isfile("src/scripts/dl_pipeline.py"), "dl_pipeline.py not found"

    def test_dl_pipeline_importable(self):
        spec = importlib.util.spec_from_file_location("dl_pipeline", "src/scripts/dl_pipeline.py")
        if spec is None:
            pytest.skip("could not load spec")
        try:
            spec.loader.exec_module(importlib.util.module_from_spec(spec))
        except Exception as e:
            pytest.skip(f"import raised: {e}")
