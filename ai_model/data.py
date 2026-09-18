"""Read dialogue data and form batches without depending on training code."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from datasets import DatasetDict
    from .tokenizer import Tokenizer
from jieba import cut


DEFAULT_DIALOG_FIELD = "dialog"
LOCAL_SPLITS = frozenset({"train", "validation", "test", "unknown"})


class DataSource:
    """Load LCCC dialogs or a local finite-domain chat dataset."""

    def __init__(
        self,
        dataset: str,
        max_dialogs: int = 10_000,
        dataset_config: Optional[str] = None,
        dialog_field: str = DEFAULT_DIALOG_FIELD,
    ):
        if not isinstance(dataset, str) or not dataset.strip():
            raise ValueError("dataset must be a non-empty string")
        if max_dialogs <= 0:
            raise ValueError("max_dialogs must be greater than 0")
        if dataset_config is not None:
            if not isinstance(dataset_config, str) or not dataset_config.strip():
                raise ValueError("dataset_config must be a non-empty string")
        if not isinstance(dialog_field, str) or not dialog_field.strip():
            raise ValueError("dialog_field must be a non-empty string")

        self.dataset = dataset
        self.dataset_config = dataset_config
        self.dialog_field = dialog_field
        self._ds: Optional[DatasetDict] = None
        self.max_dialogs = max_dialogs
        self.dataset_path = self._resolve_dataset_path(self.dataset)

    @staticmethod
    def _resolve_dataset_path(dataset: str) -> Optional[Path]:
        candidate = Path(dataset).expanduser()
        candidates = [candidate]

        if not candidate.is_absolute():
            candidates = [
                Path.cwd() / candidate,
            ]

        for path in candidates:
            if path.is_file():
                if path.suffix.lower() != ".json":
                    raise ValueError(
                        f"Local chat dataset must be a JSON file: {path}"
                    )
                return path.resolve()

        if candidate.suffix.lower() == ".json":
            searched_paths = ", ".join(str(path) for path in candidates)
            raise FileNotFoundError(
                f"Local chat dataset does not exist. Searched: {searched_paths}"
            )

        return None

    @property
    def is_local_dataset(self) -> bool:
        return self.dataset_path is not None

    @property
    def ds(self) -> DatasetDict:
        if self.is_local_dataset:
            raise TypeError(
                "A local chat dataset is not a Hugging Face DatasetDict. "
                "Use get_records() or get_pairs() instead."
            )
        if self._ds is None:
            # Remote dependencies and network access are only needed here.
            from datasets import DatasetDict, load_dataset

            if self.dataset_config is None:
                loaded_dataset = load_dataset(self.dataset)
            else:
                loaded_dataset = load_dataset(
                    self.dataset,
                    self.dataset_config,
                )
            if not isinstance(loaded_dataset, DatasetDict):
                raise TypeError(
                    "Expected load_dataset() to return a DatasetDict, got "
                    f"{type(loaded_dataset).__name__}"
                )
            self._ds = loaded_dataset
        return self._ds

    @staticmethod
    def clean_turn(text: str) -> str:
        return "".join(str(text).split())

    @lru_cache(maxsize=1)
    def _load_local_payload(self) -> dict[str, Any]:
        if self.dataset_path is None:
            raise TypeError("The configured dataset is not a local JSON file")

        try:
            with self.dataset_path.open(mode="r", encoding="utf-8") as file:
                payload = json.load(file)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON in local chat dataset {self.dataset_path}: {error}"
            ) from error

        if isinstance(payload, list):
            payload = {"train": payload}
        if not isinstance(payload, dict):
            raise TypeError(
                "Local chat dataset must be a JSON object with train, "
                "validation, test, and optional unknown lists"
            )
        if "train" not in payload:
            raise ValueError("Local chat dataset must contain a train list")

        return payload

    @staticmethod
    def _read_required_text(
        record: dict[str, Any],
        field: str,
        split: str,
        index: int,
    ) -> str:
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"{split}[{index}].{field} must be a non-empty string"
            )
        return value.strip()

    @lru_cache(maxsize=8)
    def get_records(self, split: str = "train") -> list[dict[str, Any]]:
        if not self.is_local_dataset:
            raise TypeError("get_records() is only available for local JSON datasets")
        if split not in LOCAL_SPLITS:
            raise ValueError(f"Unsupported dataset split: {split}")

        payload = self._load_local_payload()
        raw_records = payload.get(split, [])
        if not isinstance(raw_records, list):
            raise TypeError(f"Local chat dataset field {split} must be a list")
        if split == "train" and not raw_records:
            raise ValueError("Local chat dataset train list cannot be empty")

        records: list[dict[str, Any]] = []
        record_ids: set[str] = set()

        for index, raw_record in enumerate(raw_records):
            if split == "unknown" and isinstance(raw_record, str):
                raw_record = {"question": raw_record}
            if not isinstance(raw_record, dict):
                raise TypeError(f"{split}[{index}] must be an object")

            record = dict(raw_record)
            record["question"] = self.clean_turn(
                self._read_required_text(record, "question", split, index)
            )

            if split != "unknown":
                record["answer"] = self.clean_turn(
                    self._read_required_text(record, "answer", split, index)
                )

                intent = record.get("intent")
                if intent is not None:
                    if not isinstance(intent, str) or not intent.strip():
                        raise ValueError(
                            f"{split}[{index}].intent must be a non-empty string"
                        )
                    record["intent"] = intent.strip()

            record_id = record.get("record_id")
            if record_id is not None:
                if not isinstance(record_id, str) or not record_id.strip():
                    raise ValueError(
                        f"{split}[{index}].record_id must be a non-empty string"
                    )
                record_id = record_id.strip()
                if record_id in record_ids:
                    raise ValueError(
                        f"Duplicate record_id in {split}: {record_id}"
                    )
                record_ids.add(record_id)
                record["record_id"] = record_id

            answer_id = record.get("answer_id")
            if answer_id is not None:
                if split == "unknown":
                    raise ValueError(
                        f"{split}[{index}] cannot contain answer_id"
                    )
                if not isinstance(answer_id, str) or not answer_id.strip():
                    raise ValueError(
                        f"{split}[{index}].answer_id must be a non-empty string"
                    )
                record["answer_id"] = answer_id.strip()

            records.append(record)

        return records

    @lru_cache(maxsize=8)
    def get_dialogs(self, split: str = "train") -> list[list[str]]:
        if split not in LOCAL_SPLITS - {"unknown"}:
            raise ValueError(f"Unsupported dialog split: {split}")

        if self.is_local_dataset:
            records = self.get_records(split)[: self.max_dialogs]
            return [
                [record["question"], record["answer"]]
                for record in records
            ]

        dataset_dict = self.ds
        if split not in dataset_dict:
            raise ValueError(
                f"Dataset {self.dataset!r} has no {split!r} split. "
                f"Available splits: {list(dataset_dict.keys())}"
            )
        dataset_split = dataset_dict[split]
        dialog_field = self.dialog_field
        if dialog_field not in dataset_split.column_names:
            raise ValueError(
                f"Dataset field {dialog_field!r} was not found. "
                f"Available fields: {dataset_split.column_names}"
            )

        sample_count = min(self.max_dialogs, len(dataset_split))
        dialogs: list[list[str]] = []

        for row_index, row in enumerate(
            dataset_split.select(range(sample_count))
        ):
            raw_turns = row[dialog_field]
            if not isinstance(raw_turns, list):
                raise TypeError(
                    f"{split}[{row_index}].{dialog_field} must be a list"
                )
            turns: list[str] = []
            for turn_index, turn in enumerate(raw_turns):
                if not isinstance(turn, str):
                    raise TypeError(
                        f"{split}[{row_index}].{dialog_field}"
                        f"[{turn_index}] must be a string"
                    )
                cleaned_turn = self.clean_turn(turn)
                if cleaned_turn:
                    turns.append(cleaned_turn)
            if turns:
                dialogs.append(turns)

        if not dialogs:
            raise ValueError("No dialogs were found in the dataset")

        return dialogs

    @lru_cache(maxsize=8)
    def get_pairs(self, split: str = "train") -> list[tuple[str, str]]:
        if split == "unknown":
            raise ValueError("The unknown split has no answers to form pairs")
        if self.is_local_dataset:
            pairs = [
                (record["question"], record["answer"])
                for record in self.get_records(split)[: self.max_dialogs]
            ]
            if not pairs:
                raise ValueError(
                    f"No {split} pairs were found in the local dataset"
                )
            return pairs

        pairs: list[tuple[str, str]] = []

        for turns in self.get_dialogs(split):
            pairs.extend(zip(turns[:-1], turns[1:]))

        if not pairs:
            raise ValueError(
                f"No adjacent dialog pairs were found in the {split} split"
            )

        return pairs

    @lru_cache(maxsize=1)
    def load_word_data(self) -> list[list[str]]:
        if self.is_local_dataset:
            return [
                list(cut(text))
                for pair in self.get_pairs()
                for text in pair
            ]

        return [
            list(cut(turn))
            for dialog in self.get_dialogs()
            for turn in dialog
        ]


class TrainingDataset(Dataset):
    """Pre-encode every training pair once and batch the cached tensors."""

    def __init__(self, pairs: list[tuple[str, str]], tokenizer: Tokenizer):
        self.pad_id = tokenizer.pad_id
        self.examples = [
            (
                torch.tensor(tokenizer.get_ids(question), dtype=torch.long),
                torch.tensor(tokenizer.get_ids(answer), dtype=torch.long),
            )
            for question, answer in pairs
        ]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.examples[index]

    def collate_fn(
        self,
        batch: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        question_ids, answer_ids = zip(*batch)
        padded_question_ids = pad_sequence(
            question_ids,
            batch_first=True,
            padding_value=self.pad_id,
        )
        padded_answer_ids = pad_sequence(
            answer_ids,
            batch_first=True,
            padding_value=self.pad_id,
        )
        question_lengths = torch.tensor(
            [len(ids) for ids in question_ids],
            dtype=torch.long,
        )
        return padded_question_ids, padded_answer_ids, question_lengths

    def __len__(self) -> int:
        return len(self.examples)