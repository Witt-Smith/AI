"""Jieba segmentation and stable token IDs, composed with an embedding store."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Union

import torch
from jieba import cut

from .embeddings import EmbeddingStore


class Tokenizer:
    """Segment text and keep token IDs aligned with training artifacts."""

    SPECIAL_TOKEN_IDS = {
        "<PAD>": 0,
        "<UNK>": 1,
        "<BOS>": 2,
        "<EOS>": 3,
    }

    def __init__(
        self,
        store: EmbeddingStore,
        vocabulary: dict[str, int],
        vocabulary_path: Path,
        max_sequence_length: int = 64,
    ):
        if max_sequence_length < 3:
            raise ValueError("max_sequence_length must be at least 3")
        self.validate_vocabulary(vocabulary)
        self.store = store
        self.max_sequence_length = max_sequence_length
        self.vocabulary_path = Path(vocabulary_path)
        self.token_to_id = dict(vocabulary)
        self.id_to_token = {
            token_id: token for token, token_id in self.token_to_id.items()
        }
        self.pad_id = self.SPECIAL_TOKEN_IDS["<PAD>"]
        self.unk_id = self.SPECIAL_TOKEN_IDS["<UNK>"]
        self.bos_id = self.SPECIAL_TOKEN_IDS["<BOS>"]
        self.eos_id = self.SPECIAL_TOKEN_IDS["<EOS>"]
        self._ignored_decode_ids = frozenset({self.pad_id, self.bos_id})

    @classmethod
    def prepare(
        cls,
        store: EmbeddingStore,
        vocabulary_path: Path,
        max_sequence_length: int = 64,
        allow_vocabulary_updates: bool = True,
    ) -> Tokenizer:
        """Create IDs once; an existing vocabulary is always reused unchanged."""
        path = Path(vocabulary_path)
        if path.exists() or not allow_vocabulary_updates:
            return cls.load(store, path, max_sequence_length)
        vocabulary = dict(cls.SPECIAL_TOKEN_IDS)
        for token in sorted(store.word2vec.wv.key_to_index):
            if token not in vocabulary:
                vocabulary[token] = len(vocabulary)
        tokenizer = cls(store, vocabulary, path, max_sequence_length)
        tokenizer._write_vocabulary(vocabulary)
        return tokenizer

    @classmethod
    def load(
        cls,
        store: EmbeddingStore,
        vocabulary_path: Path,
        max_sequence_length: int = 64,
    ) -> Tokenizer:
        """Read fixed IDs without creating artifacts or accessing a dataset."""
        path = Path(vocabulary_path)
        if not path.is_file():
            raise FileNotFoundError(f"Vocabulary file does not exist: {path}")
        if path.stat().st_size == 0:
            raise ValueError(f"Vocabulary file is empty: {path}")
        with path.open(mode="r", encoding="utf-8") as file:
            vocabulary = json.load(file)
        return cls(store, vocabulary, path, max_sequence_length)

    def get_vector(self, vocabulary: dict[str, int]) -> torch.Tensor:
        # Keep this call at Train construction time: it consumes torch RNG state.
        return self.store.get_vector(vocabulary)

    def word_segmentation(self, text: str) -> list[str]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return list(cut(text.strip()))

    @classmethod
    def validate_vocabulary(cls, vocabulary: dict[str, int]) -> None:
        if not isinstance(vocabulary, dict) or not vocabulary:
            raise ValueError("Vocabulary must be a non-empty JSON object")

        for token, token_id in vocabulary.items():
            if not isinstance(token, str):
                raise TypeError("Vocabulary tokens must be strings")
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise TypeError(f"Vocabulary ID for {token!r} must be an integer")

        for token, expected_id in cls.SPECIAL_TOKEN_IDS.items():
            actual_id = vocabulary.get(token)
            if actual_id != expected_id:
                raise ValueError(
                    f"Invalid special-token ID for {token}: "
                    f"expected {expected_id}, got {actual_id}"
                )

        token_ids = sorted(vocabulary.values())
        if token_ids != list(range(len(vocabulary))):
            raise ValueError("Vocabulary IDs must be unique and contiguous")

    def _write_vocabulary(self, vocabulary: dict[str, int]) -> None:
        self.vocabulary_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.vocabulary_path.with_suffix(
            self.vocabulary_path.suffix + ".tmp"
        )
        with temporary_path.open(mode="w", encoding="utf-8") as file:
            json.dump(vocabulary, file, ensure_ascii=False)
        temporary_path.replace(self.vocabulary_path)

    def get_ids(self, text: str) -> list[int]:
        token_to_id = self.token_to_id
        token_ids = [
            token_to_id.get(token, self.unk_id)
            for token in self.word_segmentation(text)[
                : self.max_sequence_length - 2
            ]
        ]
        return [
            self.bos_id,
            *token_ids,
            self.eos_id,
        ]

    def decode(
        self,
        token_ids: Union[Iterable[int], torch.Tensor],
    ) -> str:
        if isinstance(token_ids, torch.Tensor):
            decoded_ids = token_ids.detach().cpu().flatten().tolist()
        else:
            decoded_ids = list(token_ids)

        tokens: list[str] = []

        for token_id in decoded_ids:
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise TypeError("token_ids must contain integers")
            if token_id == self.eos_id:
                break
            if token_id not in self._ignored_decode_ids:
                tokens.append(self.id_to_token.get(token_id, "<UNK>"))

        return "".join(tokens)
