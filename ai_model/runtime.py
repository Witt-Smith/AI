"""Composition root: explicit preparation, training, restoration and inference."""
from dataclasses import fields, replace
from typing import Any, Optional
import json
import math
import warnings

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader

from .checkpoints import (RunPaths, artifact_fingerprints, check_checkpoint_origin, check_manifest,
                          load_checkpoint, read_manifest, require_artifacts, resolve_checkpoint,
                          validate_checkpoint, write_run_record)
from .config import AppConfig, DataConfig, TrainConfig
from .data import DataSource, TrainingDataset
from .embeddings import EmbeddingStore
from .inference import ChatSession, resolve_device
from .tokenizer import Tokenizer
from .training import Train


def restore_run_config(config: AppConfig, manifest: Optional[dict[str, Any]]) -> AppConfig:
    if manifest is None:
        return config
    saved = manifest["config"]
    data = {}
    for field in fields(DataConfig):
        key = field.name
        previous = saved["data"][key]
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
            train[key] = saved["train"][key]
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


def run_train(config: AppConfig) -> None:
    paths = RunPaths(config.output_dir)
    checkpoint = resolve_checkpoint(config.train.resume_from, config.train.no_resume, paths.checkpoint_dir)
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
    source = DataSource(config.data.dataset_name, max_dialogs=config.data.max_dialogs,
                        dataset_config=config.data.dataset_config, dialog_field=config.data.dialog_field)
    pairs = source.get_pairs()
    paths.prepare()
    store = EmbeddingStore.load(paths.word2vec_path) if all(present) else EmbeddingStore.prepare(source, paths.word2vec_path)
    tokenizer = Tokenizer.prepare(store, paths.vocabulary_path, config.data.max_sequence_length,
                                  allow_vocabulary_updates=not all(present))
    dataset = TrainingDataset(pairs, tokenizer)
    loader = DataLoader(dataset, batch_size=config.train.batch_size, shuffle=True,
                        num_workers=config.train.num_workers, collate_fn=dataset.collate_fn,
                        persistent_workers=config.train.num_workers > 0)
    model = Train(vocab_size=len(tokenizer.token_to_id), pad_id=tokenizer.pad_id,
                  tokenizer=tokenizer, learning_rate=config.train.learning_rate)
    if payload is not None:
        validate_checkpoint(payload, model)
    hashes = artifact_fingerprints(paths)
    record = write_run_record(paths, config, hashes, checkpoint)
    logger = CSVLogger(str(paths.log_dir), name="train")
    callback = ModelCheckpoint(dirpath=paths.checkpoint_dir, filename="epoch-{epoch:06d}",
                               save_last=True, save_top_k=0,
                               every_n_epochs=config.train.checkpoint_every_n_epochs,
                               save_on_exception=True, auto_insert_metric_name=False)
    trainer = L.Trainer(max_epochs=config.train.max_epochs, max_time=config.train.max_time,
                        accelerator=config.train.accelerator, devices=config.train.devices,
                        precision=config.train.precision, logger=logger, callbacks=[callback],
                        log_every_n_steps=config.train.log_every_n_steps,
                        gradient_clip_val=config.train.gradient_clip_val, deterministic="warn",
                        fast_dev_run=config.train.fast_dev_run, default_root_dir=paths.output_dir)
    print(json.dumps({"config": config.to_dict(), "training_pairs": len(dataset),
                      "vocabulary_size": len(tokenizer.token_to_id),
                      "resume_from": str(checkpoint) if checkpoint else None,
                      "run_record": str(record)}, ensure_ascii=False, indent=2))
    trainer.fit(model, train_dataloaders=loader, ckpt_path=checkpoint)


def run_chat(config: AppConfig) -> None:
    paths = RunPaths(config.output_dir)
    checkpoint = resolve_checkpoint(config.chat.checkpoint, False, paths.checkpoint_dir)
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
    print(f"checkpoint={checkpoint}\ndevice={next(model.parameters()).device}")
    ChatSession(model, tokenizer).begin_chat(config.chat.max_new_tokens, config.chat.typing_delay)
