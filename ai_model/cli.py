"""CLI owns parsing; runtime owns execution. Help never loads Torch or data."""
import argparse
from pathlib import Path
from typing import Optional

from .config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ai_model",
        description="Train and use your own GRU dialogue model.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("train", "chat"):
        sub = commands.add_parser(command, argument_default=argparse.SUPPRESS)
        sub.add_argument("--config", type=Path, help="YAML file; explicit flags override its fields")
        sub.add_argument("--output-dir", type=Path, help="One persistent experiment directory")
        sub.add_argument("--max-sequence-length", type=int)
        if command == "train":
            for flag in ("dataset-name", "dataset-config", "dialog-field", "accelerator", "precision", "max-time"):
                sub.add_argument(f"--{flag}")
            for flag in ("max-dialogs", "batch-size", "num-workers", "seed", "checkpoint-every-n-epochs", "log-every-n-steps"):
                sub.add_argument(f"--{flag}", type=int)
            sub.add_argument("--devices", type=lambda value: int(value) if value.isdigit() else value)
            sub.add_argument("--learning-rate", type=float)
            sub.add_argument("--gradient-clip-val", type=float)
            epochs = sub.add_mutually_exclusive_group()
            epochs.add_argument("--max-epochs", type=int, help="Total target epoch, including restored epochs")
            epochs.add_argument("--continuous", action="store_true")
            resume = sub.add_mutually_exclusive_group()
            resume.add_argument("--resume-from", type=Path)
            resume.add_argument("--no-resume", action="store_true", help="Fresh weights; refuses a directory containing checkpoints")
            sub.add_argument("--fast-dev-run", action="store_true", help="One batch only; does NOT verify checkpoint saving")
        else:
            sub.add_argument("--checkpoint", type=Path)
            sub.add_argument("--device")
            sub.add_argument("--max-new-tokens", type=int)
            sub.add_argument("--typing-delay", type=float)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = vars(build_parser().parse_args(argv))
    command = args.pop("command")
    config_path = args.pop("config", None)
    if args.pop("continuous", False):
        args["max_epochs"] = -1
    data_fields = {"dataset_name", "dataset_config", "dialog_field", "max_dialogs", "max_sequence_length"}
    overrides = {key if key == "output_dir" else f"{'data' if key in data_fields else command}.{key}": value
                 for key, value in args.items()}
    try:
        config = load_config(command, config_path, overrides)
        from .runtime import run_chat, run_train
        if command == "train":
            run_train(config)
        else:
            run_chat(config)
    except (ValueError, TypeError, OSError) as error:
        print(f"Error: {error}", file=__import__("sys").stderr)
        return 2
    return 0
