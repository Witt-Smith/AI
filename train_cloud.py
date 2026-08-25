import argparse
from pathlib import Path
from typing import Optional, Union

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader

from color import GREEN, RESET
from config import Config, PROJECT_OUTPUT_DIR, TrainingConfig
from data_provider import DataProvider, LoadingMethod
from tokenizer import Tokenizer
from train import Train, TrainingDataset


def parse_devices(value: str) -> Union[str, int]:
    return int(value) if value.isdigit() else value


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for a resumable training job."""
    parser = argparse.ArgumentParser(
        description="Run resumable AI-MODEL training for a local or cloud job."
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_OUTPUT_DIR)
    parser.add_argument("--dataset-name", default="silver/lccc")
    parser.add_argument("--dataset-config", default="base")
    parser.add_argument("--dialog-field", default="dialog")
    parser.add_argument("--max-dialogs", type=int, default=10_000)
    parser.add_argument("--max-sequence-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=700)
    parser.add_argument("--num-workers", type=int, default=400)
    parser.add_argument("--max-epochs", type=int, default=1_000)
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Train without an epoch limit. Stop the cloud job to end training.",
    )
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-every-n-epochs", type=int, default=100)
    parser.add_argument("--log-every-n-steps", type=int, default=100)
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", type=parse_devices, default="auto")
    parser.add_argument("--precision", default="32-true")
    parser.add_argument("--max-time", default=None)
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Resume from this checkpoint instead of auto-detecting one.",
    )
    resume_group.add_argument(
        "--no-resume",
        action="store_true",
        help=(
            "Start new network weights but reuse compatible vocabulary and "
            "Word2Vec artifacts already present in output-dir."
        ),
    )
    parser.add_argument("--fast-dev-run", action="store_true")
    return parser


def newest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """Prefer last.ckpt, otherwise return the newest checkpoint file."""
    last_checkpoint = checkpoint_dir / "last.ckpt"
    if last_checkpoint.is_file():
        return last_checkpoint
    return max(
        (path for path in checkpoint_dir.glob("*.ckpt") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        default=None,
    )


def resolve_checkpoint(
    resume_from: Optional[Path],
    no_resume: bool,
    checkpoint_dir: Path,
) -> Optional[Path]:
    """Resolve an explicit checkpoint or fall back to automatic resume."""
    if resume_from is not None:
        checkpoint = resume_from.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        return checkpoint
    if no_resume:
        return None
    return newest_checkpoint(checkpoint_dir)


def main() -> None:
    args = build_parser().parse_args()
    max_epochs = -1 if args.continuous else args.max_epochs
    output_dir = args.output_dir.expanduser().resolve()
    training_config = TrainingConfig(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dialog_field=args.dialog_field,
        max_dialogs=args.max_dialogs,
        max_sequence_length=args.max_sequence_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_epochs=max_epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        checkpoint_every_n_epochs=args.checkpoint_every_n_epochs,
        log_every_n_steps=args.log_every_n_steps,
        gradient_clip_val=args.gradient_clip_val,
    )
    config = Config(
        output_dir=output_dir,
        training=training_config,
    )
    config.prepare()

    checkpoint_dir = config.checkpoint_dir
    log_dir = config.log_dir
    vocabulary_path = config.vocabulary_path
    word2vec_path = config.word2vec_path
    checkpoint_path = resolve_checkpoint(
        args.resume_from,
        args.no_resume,
        checkpoint_dir,
    )

    if checkpoint_path is not None:
        for artifact_path in (vocabulary_path, word2vec_path):
            if not artifact_path.is_file():
                raise FileNotFoundError(
                    "A checkpoint exists but a required training artifact is missing: "
                    f"{artifact_path}"
                )

    L.seed_everything(training_config.seed, workers=True)
    data_provider = DataProvider(
        training_config.dataset_name,
        LoadingMethod.PATH,
        max_dialogs=training_config.max_dialogs,
        dataset_config=training_config.dataset_config,
        dialog_field=training_config.dialog_field,
    )
    tokenizer = Tokenizer(
        data_provider,
        max_sequence_length=training_config.max_sequence_length,
        vocabulary_path=vocabulary_path,
        word2vec_path=word2vec_path,
        allow_vocabulary_updates=checkpoint_path is None,
    )
    vocabulary_size = len(tokenizer.token_to_id)

    train_dataset = TrainingDataset(tokenizer)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
        collate_fn=train_dataset.collate_fn,
        persistent_workers=training_config.num_workers > 0,
    )
    training_model = Train(
        vocab_size=vocabulary_size,
        pad_id=tokenizer.pad_id,
        tokenizer=tokenizer,
        learning_rate=training_config.learning_rate,
    )

    csv_logger = CSVLogger(str(log_dir), name="train")
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="epoch-{epoch:06d}",
        save_last=True,
        save_top_k=0,
        every_n_epochs=training_config.checkpoint_every_n_epochs,
        save_on_exception=True,
        auto_insert_metric_name=False,
    )
    trainer = L.Trainer(
        max_epochs=training_config.max_epochs,
        max_time=args.max_time,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        logger=csv_logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=training_config.log_every_n_steps,
        gradient_clip_val=training_config.gradient_clip_val,
        deterministic="warn",
        fast_dev_run=args.fast_dev_run,
        default_root_dir=output_dir,
    )
    print(f"{GREEN}output_dir={output_dir}")
    print(f"dataset_name={training_config.dataset_name}")
    print(f"dataset_config={data_provider.dataset_config or 'default'}")
    print(f"dialog_field={training_config.dialog_field}")
    print(f"vocabulary_size={vocabulary_size}")
    print(f"word2vec={word2vec_path}")
    print(f"training_pairs={len(train_dataset)}")
    print(f"resume_from={checkpoint_path or 'none'}")
    print(f"max_epochs={training_config.max_epochs}{RESET}")
    trainer.fit(
        training_model,
        train_dataloaders=train_dataloader,
        ckpt_path=checkpoint_path,
    )


if __name__ == "__main__":
    main()
