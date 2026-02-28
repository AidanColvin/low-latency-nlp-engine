import os, subprocess, pytest

BUILD = "src/cpp/build"), "src/cpp/build")
TRAIN = os.path.join(BUILD, "train_cpp")
PREDICT = os.path.join(BUILD, "predict_cpp")
CPP_SRC = "src/cpp/src"
CPP_INC = "src/cpp/include"

class TestCppSourceFiles:
    EXPECTED_SRC = [
        "args.cpp","csv_write.cpp","dataset.cpp","folds.cpp","hashing.cpp",
        "json_write.cpp","logreg_model.cpp","metrics.cpp","model_io.cpp",
        "ngrams.cpp","sgd_trainer.cpp","sparse_vec.cpp","text_clean.cpp",
        "tokenizer.cpp","tsv_reader.cpp",
    ]
    EXPECTED_HEADERS = [
        "args.h","csv_write.h","dataset.h","folds.h","hashing.h","json_write.h",
        "logreg_model.h","metrics.h","model_io.h","ngrams.h","rows.h",
        "sgd_trainer.h","sparse_vec.h","text_clean.h","tokenizer.h","tsv_reader.h",
    ]
    def test_src_dir_exists(self):
        assert os.path.isdir(CPP_SRC)
    def test_include_dir_exists(self):
        assert os.path.isdir(CPP_INC)
    @pytest.mark.parametrize("f", EXPECTED_SRC)
    def test_src_file_exists(self, f):
        assert os.path.isfile(os.path.join(CPP_SRC, f)), f"Missing: src/cpp/src/{f}"
    @pytest.mark.parametrize("h", EXPECTED_HEADERS)
    def test_header_exists(self, h):
        assert os.path.isfile(os.path.join(CPP_INC, h)), f"Missing: src/cpp/include/{h}"
    def test_no_zero_byte_cpp(self):
        for f in os.listdir(CPP_SRC):
            if f.endswith((".cpp",".h")):
                p = os.path.join(CPP_SRC, f)
                assert os.path.getsize(p) > 0, f"{f} is 0 bytes"
    def test_cmake_exists(self):
        assert os.path.isfile("src/cpp/CMakeLists.txt")

class TestCppBinaries:
    def test_build_dir_exists(self):
        assert os.path.isdir(BUILD), f"{BUILD} missing — run cmake && make"
    def test_train_binary_exists(self):
        assert os.path.isfile(TRAIN), f"{TRAIN} not found"
    def test_predict_binary_exists(self):
        assert os.path.isfile(PREDICT), f"{PREDICT} not found"
    def test_train_executable(self):
        if not os.path.isfile(TRAIN): pytest.skip("not built")
        assert os.access(TRAIN, os.X_OK)
    def test_predict_executable(self):
        if not os.path.isfile(PREDICT): pytest.skip("not built")
        assert os.access(PREDICT, os.X_OK)

class TestCppBinariesRun:
    def _run(self, binary):
        return subprocess.run([binary,"--help"], capture_output=True, text=True, timeout=10)
    def test_train_responds(self):
        if not os.path.isfile(TRAIN): pytest.skip("not built")
        r = self._run(TRAIN)
        assert len(r.stdout + r.stderr) > 0
    def test_predict_responds(self):
        if not os.path.isfile(PREDICT): pytest.skip("not built")
        r = self._run(PREDICT)
        assert len(r.stdout + r.stderr) > 0
    def test_train_mentions_mode(self):
        if not os.path.isfile(TRAIN): pytest.skip("not built")
        r = self._run(TRAIN)
        out = (r.stdout + r.stderr).lower()
        assert "mode" in out or "train" in out or "cv" in out
    def test_train_end_to_end(self, tmp_path):
        if not os.path.isfile(TRAIN) or not os.path.isfile(PREDICT): pytest.skip("not built")
        import shutil
        train_data = tmp_path / "train.tsv"
        test_data  = tmp_path / "test.tsv"
        out_dir    = tmp_path / "out"
        out_dir.mkdir()
        rows = ["text\tlabel"] + [f"good product number {i}\t1" for i in range(30)] +                [f"bad product number {i}\t0" for i in range(30)]
        train_data.write_text("\n".join(rows))
        test_data.write_text("\n".join(["text"] + [f"test text {i}" for i in range(5)]))
        r = subprocess.run(
            [TRAIN,"--mode","fit_all","--train",str(train_data),
             "--out",str(out_dir),"--epochs","2","--lr","0.1"],
            capture_output=True, text=True, timeout=60
        )
        model = out_dir / "model.bin"
        assert model.exists(), f"model.bin not created\nstdout:{r.stdout}\nstderr:{r.stderr}"
