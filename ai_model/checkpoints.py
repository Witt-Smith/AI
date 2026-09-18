"""Experiment paths, strict selection and transparent provenance records."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from datetime import datetime, timezone
from importlib import metadata
import hashlib
import json
import uuid
import warnings


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
        for path in (self.artifact_dir, self.checkpoint_dir, self.log_dir, self.output_dir / "records"):
            path.mkdir(parents=True, exist_ok=True)


def resolve_checkpoint(explicit: Optional[Path], no_resume: bool,
                       checkpoint_dir: Path) -> Optional[Path]:
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
    expected = model.state_dict()
    saved = payload["state_dict"]
    if set(expected) != set(saved):
        raise ValueError("Checkpoint parameter names differ from the protected Train model")
    for key, tensor in expected.items():
        if not hasattr(saved[key], "shape") or tensor.shape != saved[key].shape:
            raise ValueError(f"Checkpoint shape mismatch at {key}")
    parameters = payload.get("hyper_parameters", {})
    for key in ("vocab_size", "pad_id"):
        if key in parameters and parameters[key] != model.hparams[key]:
            raise ValueError(f"Checkpoint {key} conflicts with the vocabulary")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_fingerprints(paths: RunPaths) -> dict[str, str]:
    require_artifacts(paths)
    files = [paths.vocabulary_path, *sorted(paths.artifact_dir.glob("word2vec.model*"))]
    return {path.name: sha256_file(path) for path in files if path.is_file()}


def read_manifest(paths: RunPaths) -> Optional[dict[str, Any]]:
    if not paths.manifest_path.is_file():
        return None
    with paths.manifest_path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict) or value.get("schema_version") != 1 or not isinstance(value.get("config"), dict):
        raise ValueError(f"Invalid run manifest: {paths.manifest_path}")
    if not isinstance(value.get("artifacts"), dict):
        raise ValueError("Run manifest has no artifact fingerprints")
    return value


def check_manifest(paths: RunPaths, manifest: Optional[dict[str, Any]]) -> None:
    if manifest is not None and artifact_fingerprints(paths) != manifest["artifacts"]:
        raise ValueError("Artifacts changed since this experiment was created; use its original vocabulary and Word2Vec")


def check_checkpoint_origin(checkpoint: Path, hashes: dict[str, str]) -> None:
    """An adjacent manifest checks artifact identity, not checkpoint content authenticity."""
    origin = RunPaths(checkpoint.parent.parent)
    manifest = read_manifest(origin) if checkpoint.parent.name == "checkpoints" else None
    if manifest is None:
        warnings.warn("Legacy/unbound checkpoint: names, shapes and token IDs can be checked, but same-size token reordering cannot be proved compatible.", UserWarning)
    elif manifest["artifacts"] != hashes:
        raise ValueError("Selected checkpoint's experiment uses different vocabulary/Word2Vec artifacts")


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_run_record(paths: RunPaths, config: Any, hashes: dict[str, str],
                     resume_from: Optional[Path]) -> Path:
    """Keep first-run identity plus one immutable configuration record per invocation."""
    versions = {}
    for name in ("torch", "lightning", "gensim", "jieba", "datasets", "PyYAML"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    record = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
              "config": config.to_dict(), "artifacts": hashes, "dependencies": versions,
              "resume_from": str(resume_from) if resume_from else None}
    if not paths.manifest_path.exists():
        atomic_json(paths.manifest_path, record)
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8] + ".json"
    path = paths.output_dir / "records" / name
    atomic_json(path, record)
    return path
