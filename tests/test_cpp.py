import pytest
from pathlib import Path
import subprocess

BUILD = Path("src/cpp/build")
TRAIN_BIN = BUILD / "train_cpp"
PREDICT_BIN = BUILD / "predict_cpp"

if not TRAIN_BIN.exists() or not PREDICT_BIN.exists():
    pytest.skip(
        "C++ binaries not built. Build C++ to enable these tests: "
        "src/cpp/build/train_cpp and src/cpp/build/predict_cpp",
        allow_module_level=True,
    )

def test_cpp_binaries_exist():
    assert TRAIN_BIN.exists()
    assert PREDICT_BIN.exists()

def test_cpp_binaries_run_help():
    for binp in (TRAIN_BIN, PREDICT_BIN):
        r = subprocess.run([str(binp), "--help"], capture_output=True, text=True)
        assert r.returncode in (0, 1)
        assert (r.stdout + r.stderr).strip() != ""
