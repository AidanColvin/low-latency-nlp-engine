# run c++ predictor and enforce submission checkpoint
import subprocess  # run binaries
from pathlib import Path  # paths

def run_predict_cpp(test_tsv: Path, model_bin: Path, out_csv: Path, predict_bin: Path) -> None:
    """
    given test.tsv and model.bin and output csv path
    run predictor binary and require submission checkpoint
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)  # ensure stage dir exists
    subprocess.run([str(predict_bin), str(test_tsv), str(model_bin), str(out_csv)], check=True)  # run predictor
    if not out_csv.exists(): raise FileNotFoundError(f"missing checkpoint: {out_csv}")  # fail fast