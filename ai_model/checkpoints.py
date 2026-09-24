"""Experiment paths, strict selection and transparent provenance records."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional
from datetime import datetime, timezone
from importlib import metadata
import hashlib
import json
import logging
import uuid
import warnings

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunPaths:
    output_dir: Path

    @property
    def artifact_dir(self) -> Path:
        return self.output_dir / "artifacts"

    @property
    def vocabulary_path(self) -> Path:
        return self.artifact_dir / "vocabulary.json"

    @property
    def word2vec_path(self) -> Path:
        return self.artifact_dir / "word2vec.model"

    @property
    def checkpoint_dir(self) -> Path:
        return self.output_dir / "checkpoints"

    @property
    def log_dir(self) -> Path:
        return self.output_dir / "logs"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"

    def prepare(self) -> None:
        self.prepare_runtime()
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def prepare_runtime(self) -> None:
        for path in (self.checkpoint_dir, self.log_dir, self.output_dir / "records"):
            path.mkdir(parents=True, exist_ok=True)


def resolve_checkpoint(
    explicit: Optional[Path],
    no_resume: bool,
    checkpoint_dir: Path,
    prefer_best: bool = False,
) -> Optional[Path]:
    checkpoints = [path for path in checkpoint_dir.glob("*.ckpt") if path.is_file()]
    if no_resume:
        if explicit is not None:
            raise ValueError("Explicit checkpoint and no_resume are mutually exclusive")
        if checkpoints:
            raise ValueError("--no-resume refuses existing checkpoints; choose a new output directory")
        return None
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        return path
    best = checkpoint_dir / "best.ckpt"
    if prefer_best and best.is_file():
        return best
    last = checkpoint_dir / "last.ckpt"
    if last.is_file():
        return last
    return max(checkpoints, key=lambda path: (path.stat().st_mtime_ns, path.name), default=None)


def require_artifacts(paths: RunPaths) -> None:
    for path in (paths.vocabulary_path, paths.word2vec_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Required artifact is missing or empty: {path}")


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load a user's own Lightning checkpoint; do not try another on failure."""
    import torch
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise ValueError(f"Cannot read selected checkpoint {path}: {error}") from error
    if not isinstance(value, dict) or not isinstance(value.get("state_dict"), dict):
        raise ValueError(f"Checkpoint has no valid state_dict: {path}")
    return value


def validate_checkpoint(payload: dict[str, Any], model: Any) -> None:
    import torch

    expected = model.state_dict()
    saved = payload["state_dict"]
    if set(expected) != set(saved):
        raise ValueError("Checkpoint parameter names differ from the current Train model")
    for key, tensor in expected.items():
        saved_tensor = saved[key]
        if not isinstance(saved_tensor, torch.Tensor) or tensor.shape != saved_tensor.shape:
            raise ValueError(f"Checkpoint shape mismatch at {key}")
        if tensor.dtype != saved_tensor.dtype:
            raise ValueError(f"Checkpoint dtype mismatch at {key}")
        if saved_tensor.is_floating_point() and not torch.isfinite(saved_tensor).all().item():
            raise ValueError(f"Checkpoint contains non-finite values at {key}")
    parameters = payload.get("hyper_parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError("Checkpoint hyper_parameters must be a mapping")
    for key in ("vocab_size", "pad_id"):
        if key in parameters and parameters[key] != model.hparams[key]:
            raise ValueError(f"Checkpoint {key} conflicts with the vocabulary")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def artifact_fingerprints(paths: RunPaths) -> dict[str, str]:
    require_artifacts(paths)
    files = [paths.vocabulary_path, *sorted(paths.artifact_dir.glob("word2vec.model*"))]
    return {path.name: sha256_file(path) for path in files if path.is_file()}


def dialogue_pair_fingerprint(
    pairs: Iterable[tuple[str, str]],
) -> dict[str, Any]:
    """Hash normalized pairs in order so resume cannot silently change data."""
    digest = hashlib.sha256()
    pair_count = 0
    for question, answer in pairs:
        encoded = json.dumps(
            [question, answer],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
        pair_count += 1
    return {"pair_count": pair_count, "sha256": digest.hexdigest()}


def read_manifest(paths: RunPaths) -> Optional[dict[str, Any]]:
    if not paths.manifest_path.is_file():
        return None
    with paths.manifest_path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict) or value.get("schema_version") not in {1, 2}:
        raise ValueError(f"Invalid run manifest: {paths.manifest_path}")
    config = value.get("config")
    if not isinstance(config, dict):
        raise ValueError("Run manifest has no valid config")
    for section in ("data", "train", "chat"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Run manifest config has no valid {section} section")
    artifacts = value.get("artifacts")
    if (
        not isinstance(artifacts, dict)
        or not artifacts
        or any(
            not isinstance(name, str) or not _is_sha256(fingerprint)
            for name, fingerprint in artifacts.items()
        )
    ):
        raise ValueError("Run manifest has no valid artifact fingerprints")
    if value["schema_version"] == 2:
        training_data = value.get("training_data")
        if not isinstance(training_data, dict) or "train" not in training_data:
            raise ValueError("Run manifest has no valid training-data fingerprints")
        for split, fingerprint in training_data.items():
            if (
                not isinstance(split, str)
                or not isinstance(fingerprint, dict)
                or isinstance(fingerprint.get("pair_count"), bool)
                or not isinstance(fingerprint.get("pair_count"), int)
                or fingerprint["pair_count"] < 0
                or not _is_sha256(fingerprint.get("sha256"))
            ):
                raise ValueError("Run manifest has an invalid training-data fingerprint")
    return value


def check_manifest(paths: RunPaths, manifest: Optional[dict[str, Any]]) -> None:
    if manifest is not None and artifact_fingerprints(paths) != manifest["artifacts"]:
        raise ValueError("Artifacts changed since this experiment was created; use its original vocabulary and Word2Vec")


def check_training_data(
    manifest: Optional[dict[str, Any]],
    training_data: dict[str, dict[str, Any]],
) -> None:
    if manifest is None:
        return
    saved = manifest.get("training_data")
    if saved is None:
        warnings.warn(
            "Legacy manifest has no training-data fingerprint; artifact compatibility "
            "can be checked, but source-data identity cannot be proved.",
            UserWarning,
        )
        return
    if saved != training_data:
        raise ValueError(
            "Training or validation data changed since this experiment was created; "
            "choose a new output directory"
        )


def check_checkpoint_origin(checkpoint: Path, hashes: dict[str, str]) -> None:
    """An adjacent manifest checks artifact identity, not checkpoint content authenticity."""
    origin = RunPaths(checkpoint.parent.parent)
    manifest = read_manifest(origin) if checkpoint.parent.name == "checkpoints" else None
    if manifest is None:
        warnings.warn("Legacy/unbound checkpoint: names, shapes and token IDs can be checked, but same-size token reordering cannot be proved compatible.", UserWarning)
    elif manifest["artifacts"] != hashes:
        raise ValueError("Selected checkpoint's experiment uses different vocabulary/Word2Vec artifacts")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_run_record(
    paths: RunPaths,
    config: Any,
    hashes: dict[str, str],
    training_data: dict[str, dict[str, Any]],
    resume_from: Optional[Path],
) -> Path:
    """Keep first-run identity plus one immutable configuration record per invocation."""
    versions = {}
    for name in ("torch", "lightning", "gensim", "jieba", "datasets", "PyYAML"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    record = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config.to_dict(),
        "artifacts": hashes,
        "training_data": training_data,
        "dependencies": versions,
        "resume_from": str(resume_from) if resume_from else None,
    }
    if not paths.manifest_path.exists():
        atomic_json(paths.manifest_path, record)
        LOGGER.info("Created run manifest: %s", paths.manifest_path)
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8] + ".json"
    path = paths.output_dir / "records" / name
    atomic_json(path, record)
    LOGGER.info("Wrote run record: %s", path)
    return path
