"""
Terra Sight project runner.

Single CLI entrypoint for training, evaluation, API, and UI.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

def run(cmd: list[str]) -> int:
    print("[run]", " ".join(cmd))
    return subprocess.call(cmd, cwd=ROOT)

def cmd_train(args: argparse.Namespace) -> int:
    cmd = [
        PYTHON,
        "train_isro_eo_enhanced.py",
        "--data-path",
        str(args.data_path),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--grad-accum",
        str(args.grad_accum),
        "--lr",
        str(args.lr),
    ]
    return run(cmd)

def cmd_eval(args: argparse.Namespace) -> int:
    if args.comprehensive:
        return run([PYTHON, "day5_evaluate_comprehensive.py"])
    return run([PYTHON, "day4_evaluate.py"])

def cmd_api(_: argparse.Namespace) -> int:
    return run([PYTHON, "-m", "uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"])

def cmd_streamlit(_: argparse.Namespace) -> int:
    return run([PYTHON, "-m", "streamlit", "run", "streamlit_app.py", "--server.port", "8501"])

def cmd_check(_: argparse.Namespace) -> int:
    return run([PYTHON, "-m", "py_compile", "api_server.py", "train_isro_eo_enhanced.py", "day4_evaluate.py"])

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Terra Sight project runner")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Run enhanced ISRO EO training")
    p_train.add_argument("--data-path", default="data/training/training_data.json")
    p_train.add_argument("--epochs", type=int, default=2)
    p_train.add_argument("--batch-size", type=int, default=1)
    p_train.add_argument("--grad-accum", type=int, default=4)
    p_train.add_argument("--lr", type=float, default=2e-5)
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("eval", help="Run evaluation")
    p_eval.add_argument("--comprehensive", action="store_true", help="Run day5 comprehensive evaluation")
    p_eval.set_defaults(func=cmd_eval)

    p_api = sub.add_parser("api", help="Start FastAPI backend")
    p_api.set_defaults(func=cmd_api)

    p_streamlit = sub.add_parser("streamlit", help="Start Streamlit frontend")
    p_streamlit.set_defaults(func=cmd_streamlit)

    p_check = sub.add_parser("check", help="Compile-check core scripts")
    p_check.set_defaults(func=cmd_check)

    return parser

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)

if __name__ == "__main__":
    raise SystemExit(main())
