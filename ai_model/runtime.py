"""Composition root: explicit preparation, training, restoration and inference."""
from dataclasses import fields, replace
from typing import Any, Optional
import json
import logging
import math
import shutil
import uuid
import warnings

import lightning as L
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader

from .checkpoints import (
    RunPaths,
    artifact_fingerprints,
    check_checkpoint_origin,
    check_manifest,
    check_training_data,
    dialogue_pair_fingerprint,
    load_checkpoint,
    read_manifest,
    require_artifacts,
    resolve_checkpoint,
    validate_checkpoint,
    write_run_record,
)
from .config import AppConfig, DataConfig, TrainConfig
from .data import DataSource, TrainingDataset
from .embeddings import EmbeddingStore
from .inference import ChatSession, resolve_device
from .logging_config import configure_logging
from .tokenizer import Tokenizer
from .training import Train


LOGGER = logging.getLogger(__name__)


def restore_run_config(config: AppConfig, manifest: Optional[dict[str, Any]]) -> AppConfig:
    if manifest is None:
        return config
    saved = manifest["config"]
    saved_data = saved["data"]
    saved_train = saved["train"]
    data = {}
    for field in fields(DataConfig):
        key = field.name
        if key in saved_data:
            previous = saved_data[key]
        elif key in {
            "dataset_revision",
            "validation_dialogs",
            "max_vocabulary_size",
        }:
            previous = getattr(DataConfig(), key)
        else:
            raise ValueError(f"Run manifest is missing data.{key}")
        requested = getattr(config.data, key)
        if f"data.{key}" in config.explicit:
            if requested != previous and (config.command == "train" or key == "max_sequence_length"):
                raise ValueError(f"data.{key} conflicts with this experiment; choose a new output directory")
            data[key] = requested
        else:
            data[key] = previous
    train = {}
    for field in fields(TrainConfig):
        key = field.name
        if key in {"resume_from", "no_resume", "fast_dev_run"} or f"train.{key}" in config.explicit:
            train[key] = getattr(config.train, key)
        else:
            if key not in saved_train:
                raise ValueError(f"Run manifest is missing train.{key}")
            train[key] = saved_train[key]
    result = replace(config, data=DataConfig(**data), train=TrainConfig(**train))
    result.validate()
    return result


def checkpoint_learning_rate(payload: dict[str, Any]) -> float:
    try:
        rates = [group["lr"] for optimizer in payload["optimizer_states"] for group in optimizer["param_groups"]]
        if len(rates) != 1 or not math.isfinite(rates[0]) or rates[0] <= 0:
            raise ValueError("Expected the original Adam optimizer with one positive learning rate")
        return float(rates[0])
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError("Selected checkpoint lacks complete optimizer state for resuming training") from error


def prepare_artifact_bundle(
    source: DataSource,
    paths: RunPaths,
    max_sequence_length: int,
    max_vocabulary_size: int,
) -> tuple[EmbeddingStore, Tokenizer]:
    """Build vocabulary and Word2Vec together, publishing neither on failure."""
    if paths.artifact_dir.exists():
        contents = list(paths.artifact_dir.iterdir())
        if contents:
            raise ValueError(
                "Artifact directory contains an incomplete or unknown bundle: "
                f"{paths.artifact_dir}"
            )
        paths.artifact_dir.rmdir()

    staging_dir = paths.output_dir / f".artifacts-{uuid.uuid4().hex}"
    staging_dir.mkdir(parents=False, exist_ok=False)
    staging_word2vec = staging_dir / paths.word2vec_path.name
    staging_vocabulary = staging_dir / paths.vocabulary_path.name
    try:
        store = EmbeddingStore.prepare(
            source,
            staging_word2vec,
            max_vocabulary_size=max_vocabulary_size,
        )
        tokenizer = Tokenizer.prepare(
            store,
            staging_vocabulary,
            max_sequence_length,
        )
        staging_dir.replace(paths.artifact_dir)
        store.model_path = paths.word2vec_path
        tokenizer.vocabulary_path = paths.vocabulary_path
        LOGGER.info("Published artifacts: %s", paths.artifact_dir)
        return store, tokenizer
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def make_loader(
    pairs: list[tuple[str, str]],
    tokenizer: Tokenizer,
    config: AppConfig,
    *,
    shuffle: bool,
) -> tuple[TrainingDataset, DataLoader]:
    dataset = TrainingDataset(pairs, tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=shuffle,
        num_workers=config.train.num_workers,
        collate_fn=dataset.collate_fn,
        persistent_workers=config.train.num_workers > 0,
    )
    return dataset, loader


def unknown_token_summary(dataset: TrainingDataset) -> dict[str, float]:
    return {
        "questions": round(dataset.question_unknown_ratio, 6),
        "answers": round(dataset.answer_unknown_ratio, 6),
    }


def warn_unknown_tokens(dataset: TrainingDataset, split: str) -> None:
    for side, ratio in (
        ("question", dataset.question_unknown_ratio),
        ("answer", dataset.answer_unknown_ratio),
    ):
        if ratio > 0.10:
            warnings.warn(
                f"{split} {side} unknown-token ratio is {ratio:.1%}; the model sees "
                "<UNK> instead of those tokens. Increase representative data or review "
                "tokenization, Word2Vec MIN_COUNT and the vocabulary limit before a long run.",
                UserWarning,
            )


def run_train(config: AppConfig) -> None:
    paths = RunPaths(config.output_dir)
    configure_logging(paths.log_dir)
    LOGGER.info("Training requested: output_dir=%s dataset=%s", paths.output_dir, config.data.dataset_name)
    checkpoint = resolve_checkpoint(config.train.resume_from, config.train.no_resume, paths.checkpoint_dir)
    LOGGER.info("Checkpoint selected for resume: %s", checkpoint or "none")
    manifest = read_manifest(paths)
    check_manifest(paths, manifest)
    config = restore_run_config(config, manifest)
    present = (paths.vocabulary_path.is_file(), paths.word2vec_path.is_file())
    if any(present) and not all(present):
        raise ValueError("Vocabulary and Word2Vec must be supplied together; refusing to replace partial artifacts")
    if all(present) and manifest is None:
        warnings.warn(
            "Legacy/unbound artifacts will be reused read-only. Their dimensions and token IDs "
            "can be checked, but their original dataset cannot be proved; use a new output "
            "directory if their origin is uncertain.",
            UserWarning,
        )
    payload = None
    if checkpoint is not None:
        require_artifacts(paths)
        payload = load_checkpoint(checkpoint)
        rate = checkpoint_learning_rate(payload)
        if "train.learning_rate" in config.explicit and not math.isclose(config.train.learning_rate, rate, rel_tol=1e-12):
            raise ValueError("Explicit learning_rate conflicts with the saved optimizer; use a new experiment for new weights/settings")
        config = replace(config, train=replace(config.train, learning_rate=rate))
        check_checkpoint_origin(checkpoint, artifact_fingerprints(paths))

    # Preserve original RNG order: seed -> prepare text -> Dataset -> Train.get_vector -> model layers.
    L.seed_everything(config.train.seed, workers=True)
    source = DataSource(
        config.data.dataset_name,
        max_dialogs=config.data.max_dialogs,
        dataset_config=config.data.dataset_config,
        dataset_revision=config.data.dataset_revision,
        dialog_field=config.data.dialog_field,
    )
    pairs = source.get_pairs()
    validation_pairs: list[tuple[str, str]] = []
    if config.data.validation_dialogs > 0:
        validation_source = DataSource(
            config.data.dataset_name,
            max_dialogs=config.data.validation_dialogs,
            dataset_config=config.data.dataset_config,
            dataset_revision=config.data.dataset_revision,
            dialog_field=config.data.dialog_field,
        )
        validation_pairs = validation_source.get_pairs("validation")
    LOGGER.info("Dataset ready: train_pairs=%d validation_pairs=%d", len(pairs), len(validation_pairs))

    training_data = {"train": dialogue_pair_fingerprint(pairs)}
    if validation_pairs:
        training_data["validation"] = dialogue_pair_fingerprint(validation_pairs)
    check_training_data(manifest, training_data)

    paths.prepare_runtime()
    if all(present):
        LOGGER.info("Reusing existing Word2Vec and vocabulary artifacts")
        store = EmbeddingStore.load(paths.word2vec_path)
        tokenizer = Tokenizer.prepare(
            store,
            paths.vocabulary_path,
            config.data.max_sequence_length,
            allow_vocabulary_updates=False,
        )
    else:
        LOGGER.info("Preparing new Word2Vec and vocabulary artifacts")
        store, tokenizer = prepare_artifact_bundle(
            source,
            paths,
            config.data.max_sequence_length,
            config.data.max_vocabulary_size,
        )

    dataset, loader = make_loader(pairs, tokenizer, config, shuffle=True)
    warn_unknown_tokens(dataset, "train")
    validation_loader = None
    validation_dataset = None
    if validation_pairs:
        validation_dataset, validation_loader = make_loader(
            validation_pairs,
            tokenizer,
            config,
            shuffle=False,
        )
        warn_unknown_tokens(validation_dataset, "validation")
    model = Train(vocab_size=len(tokenizer.token_to_id), pad_id=tokenizer.pad_id,
                  tokenizer=tokenizer, learning_rate=config.train.learning_rate)
    if payload is not None:
        validate_checkpoint(payload, model)
    hashes = artifact_fingerprints(paths)
    record = write_run_record(paths, config, hashes, training_data, checkpoint)
    LOGGER.info(
        "Unknown-token ratios: train=%s validation=%s",
        unknown_token_summary(dataset),
        unknown_token_summary(validation_dataset) if validation_dataset is not None else None,
    )
    logger = CSVLogger(str(paths.log_dir), name="train")
    checkpoint_options: dict[str, Any] = {
        "dirpath": paths.checkpoint_dir,
        "save_last": True,
        "every_n_epochs": config.train.checkpoint_every_n_epochs,
        "save_on_exception": True,
        "auto_insert_metric_name": False,
    }
    if validation_loader is None:
        checkpoint_options.update(filename="epoch-{epoch:06d}", save_top_k=0)
    else:
        checkpoint_options.update(
            filename="best",
            monitor="validation_loss",
            mode="min",
            save_top_k=1,
            enable_version_counter=False,
        )
    callbacks: list[Callback] = [ModelCheckpoint(**checkpoint_options)]
    trainer = L.Trainer(max_epochs=config.train.max_epochs, max_time=config.train.max_time,
                        accelerator=config.train.accelerator, devices=config.train.devices,
                        precision=config.train.precision, logger=logger, callbacks=callbacks,
                        log_every_n_steps=config.train.log_every_n_steps,
                        gradient_clip_val=config.train.gradient_clip_val, deterministic="warn",
                        fast_dev_run=config.train.fast_dev_run, default_root_dir=paths.output_dir)
    print(json.dumps({"config": config.to_dict(), "training_pairs": len(dataset),
                      "validation_pairs": len(validation_pairs),
                      "unknown_token_ratio": {
                          "train": unknown_token_summary(dataset),
                          "validation": (
                              unknown_token_summary(validation_dataset)
                              if validation_dataset is not None else None
                          ),
                      },
                      "vocabulary_size": len(tokenizer.token_to_id),
                      "resume_from": str(checkpoint) if checkpoint else None,
                      "run_record": str(record)}, ensure_ascii=False, indent=2))
    LOGGER.info("Trainer.fit starting: max_epochs=%s checkpoint=%s", config.train.max_epochs, checkpoint or "none")
    trainer.fit(
        model,
        train_dataloaders=loader,
        val_dataloaders=validation_loader,
        ckpt_path=checkpoint,
    )
    LOGGER.info("Trainer.fit finished: global_step=%d", trainer.global_step)


def run_chat(config: AppConfig) -> None:
    paths = RunPaths(config.output_dir)
    configure_logging(paths.log_dir)
    LOGGER.info("Chat requested: output_dir=%s", paths.output_dir)
    checkpoint = resolve_checkpoint(
        config.chat.checkpoint,
        False,
        paths.checkpoint_dir,
        prefer_best=True,
    )
    if checkpoint is None:
        raise FileNotFoundError(f"No checkpoint found in {paths.checkpoint_dir}")
    require_artifacts(paths)
    manifest = read_manifest(paths)
    check_manifest(paths, manifest)
    config = restore_run_config(config, manifest)
    payload = load_checkpoint(checkpoint)
    check_checkpoint_origin(checkpoint, artifact_fingerprints(paths))
    store = EmbeddingStore.load(paths.word2vec_path)
    tokenizer = Tokenizer.load(store, paths.vocabulary_path, config.data.max_sequence_length)
    model = Train(vocab_size=len(tokenizer.token_to_id), pad_id=tokenizer.pad_id, tokenizer=tokenizer,
                  learning_rate=payload.get("hyper_parameters", {}).get("learning_rate", config.train.learning_rate))
    validate_checkpoint(payload, model)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(resolve_device(config.chat.device))
    LOGGER.info("Chat model ready: checkpoint=%s device=%s", checkpoint, next(model.parameters()).device)
    print(f"checkpoint={checkpoint}\ndevice={next(model.parameters()).device}")
    ChatSession(model, tokenizer).begin_chat(config.chat.max_new_tokens, config.chat.typing_delay)
