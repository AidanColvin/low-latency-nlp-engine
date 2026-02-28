# run c++ trainer and enforce checkpoint outputs
import subprocess  # run binaries
from pathlib import Path  # paths

def run_train_cpp(train_tsv: Path, out_dir: Path, train_bin: Path) -> None:
    """
    given train.tsv path and output dir and train_cpp binary
    run training binary and require model + metrics checkpoints
    """
    out_dir.mkdir(parents=True, exist_ok=True)  # ensure stage dir exists
    subprocess.run([str(train_bin), str(train_tsv), str(out_dir)], check=True)  # run trainer

    model_path = out_dir / "model.bin"  # checkpoint
    metrics_path = out_dir / "metrics.json"  # checkpoint
    if not model_path.exists(): raise FileNotFoundError(f"missing checkpoint: {model_path}")  # fail fast
    if not metrics_path.exists(): raise FileNotFoundError(f"missing checkpoint: {metrics_path}")  # fail fast