"""One source of defaults; YAML and explicit CLI values override it."""
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Optional
import math
import re

import yaml


@dataclass(frozen=True)
class DataConfig:
    dataset_name: str = "silver/lccc"
    dataset_config: Optional[str] = "base"
    dialog_field: str = "dialog"
    max_dialogs: int = 10_000
    max_sequence_length: int = 64


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 4
    num_workers: int = 0
    max_epochs: int = 1_000
    learning_rate: float = 0.002
    seed: int = 42
    checkpoint_every_n_epochs: int = 1
    log_every_n_steps: int = 1
    gradient_clip_val: float = 1.0
    accelerator: str = "auto"
    devices: Any = "auto"
    precision: str = "32-true"
    max_time: Optional[str] = None
    resume_from: Optional[Path] = None
    no_resume: bool = False
    fast_dev_run: bool = False


@dataclass(frozen=True)
class ChatConfig:
    checkpoint: Optional[Path] = None
    device: str = "auto"
    max_new_tokens: int = 50
    typing_delay: float = 0.05


@dataclass(frozen=True)
class AppConfig:
    command: str
    output_dir: Path = field(default_factory=lambda: Path.cwd() / "runs")
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    chat: ChatConfig = field(default_factory=ChatConfig)
    explicit: frozenset[str] = field(default_factory=frozenset, repr=False)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("explicit")
        def clean(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {key: clean(item) for key, item in value.items()}
            return value
        return clean(result)

    def validate(self) -> None:
        if self.command not in {"train", "chat"}:
            raise ValueError("command must be train or chat")
        for key in ("dataset_name", "dialog_field"):
            _text(getattr(self.data, key), f"data.{key}")
        if self.data.dataset_config is not None:
            _text(self.data.dataset_config, "data.dataset_config")
        _integer(self.data.max_dialogs, "data.max_dialogs", 1)
        _integer(self.data.max_sequence_length, "data.max_sequence_length", 3)
        for key in ("batch_size", "checkpoint_every_n_epochs", "log_every_n_steps"):
            _integer(getattr(self.train, key), f"train.{key}", 1)
        _integer(self.train.num_workers, "train.num_workers", 0)
        _integer(self.train.seed, "train.seed", 0)
        if self.train.max_epochs != -1:
            _integer(self.train.max_epochs, "train.max_epochs", 1)
        elif isinstance(self.train.max_epochs, bool):
            raise ValueError("train.max_epochs must be -1 or a positive integer")
        _number(self.train.learning_rate, "train.learning_rate", positive=True)
        _number(self.train.gradient_clip_val, "train.gradient_clip_val")
        _number(self.chat.typing_delay, "chat.typing_delay")
        _integer(self.chat.max_new_tokens, "chat.max_new_tokens", 1)
        for key in ("no_resume", "fast_dev_run"):
            if not isinstance(getattr(self.train, key), bool):
                raise ValueError(f"train.{key} must be a boolean")
        if self.train.no_resume and self.train.resume_from is not None:
            raise ValueError("resume_from and no_resume are mutually exclusive")
        for value, name in ((self.train.accelerator, "train.accelerator"),
                            (self.train.precision, "train.precision"),
                            (self.chat.device, "chat.device")):
            _text(value, name)
        if self.train.devices != "auto":
            _integer(self.train.devices, "train.devices", 1)
        if self.train.max_time is not None:
            if not isinstance(self.train.max_time, str) or not re.fullmatch(r"\d+:\d{2}:\d{2}:\d{2}", self.train.max_time):
                raise ValueError("train.max_time must use DD:HH:MM:SS")
            _, hours, minutes, seconds = map(int, self.train.max_time.split(":"))
            if hours > 23 or minutes > 59 or seconds > 59:
                raise ValueError("train.max_time has an invalid clock component")


def _integer(value: Any, name: str, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _number(value: Any, name: str, positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'non-negative'}")


def _text(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _path(value: Any, base: Path) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("path must be a non-empty string")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _normalise(key: str, value: Any, base: Path) -> Any:
    if key in {"output_dir", "train.resume_from", "chat.checkpoint"}:
        return None if value is None and key != "output_dir" else _path(value, base)
    if key == "data.dataset_name" and isinstance(value, str):
        candidate = Path(value).expanduser()
        if candidate.suffix.lower() == ".json" or value.startswith(("/", "./", "../", "~")) or (base / candidate).is_file():
            return str(_path(value, base))
    return value


def load_config(command: str, config_path: Optional[Path] = None,
                overrides: Optional[dict[str, Any]] = None) -> AppConfig:
    """Resolve typed settings without creating files, loading data or training."""
    config = AppConfig(command=command)
    groups = {"data": asdict(config.data), "train": asdict(config.train), "chat": asdict(config.chat)}
    values: dict[str, Any] = {"output_dir": config.output_dir}
    values.update({f"{group}.{key}": value for group, fields in groups.items() for key, value in fields.items()})
    explicit: set[str] = set()
    if config_path is not None:
        config_path = config_path.expanduser().resolve()
        with config_path.open(encoding="utf-8") as file:
            payload = yaml.safe_load(file)
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            raise ValueError("YAML must be a mapping")
        updates: dict[str, Any] = {}
        for group, fields in payload.items():
            if group == "output_dir":
                updates[group] = fields
            elif group in groups and isinstance(fields, dict):
                updates.update({f"{group}.{key}": value for key, value in fields.items()})
            else:
                raise ValueError(f"Unknown or invalid configuration section: {group}")
        for key, value in updates.items():
            if key not in values:
                raise ValueError(f"Unknown configuration field: {key}")
            values[key] = _normalise(key, value, config_path.parent)
            explicit.add(key)
    for key, value in (overrides or {}).items():
        if key not in values:
            raise ValueError(f"Unknown command-line setting: {key}")
        values[key] = _normalise(key, value, Path.cwd())
        explicit.add(key)
    nested = {group: {key: values[f"{group}.{key}"] for key in fields} for group, fields in groups.items()}
    result = replace(config, output_dir=values["output_dir"], data=DataConfig(**nested["data"]),
                     train=TrainConfig(**nested["train"]), chat=ChatConfig(**nested["chat"]), explicit=frozenset(explicit))
    result.validate()
    return result
