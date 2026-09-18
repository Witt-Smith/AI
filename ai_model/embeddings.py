"""Explicit preparation/loading of the Word2Vec artifact."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from gensim.models import Word2Vec

if TYPE_CHECKING:
    from .data import DataSource


class EmbeddingStore:
    """Hold a ready Word2Vec model; construction never starts training."""

    VECTOR_SIZE = 128
    MIN_COUNT = 2
    EPOCHS = 1000

    def __init__(self, word2vec: Word2Vec, model_path: Path):
        self.word2vec = word2vec
        self.model_path = Path(model_path)
        if self.word2vec.vector_size != self.VECTOR_SIZE:
            raise ValueError(
                "Word2Vec vector size does not match the model embedding size: "
                f"expected {self.VECTOR_SIZE}, got {self.word2vec.vector_size}"
            )
        if not self.word2vec.wv.key_to_index:
            raise ValueError("Word2Vec vocabulary is empty")

    @classmethod
    def prepare(cls, provider: DataSource, path: Path) -> EmbeddingStore:
        """Reuse an existing artifact or explicitly train and save a new one."""
        path = Path(path)
        if path.exists():
            return cls.load(path)
        sentences = provider.load_word_data()
        token_counts = Counter(token for sentence in sentences for token in sentence)
        if not any(count >= cls.MIN_COUNT for count in token_counts.values()):
            raise ValueError(
                "Word2Vec cannot build a vocabulary: no token appears at least "
                f"{cls.MIN_COUNT} times"
            )
        model = Word2Vec(
            sentences,
            vector_size=cls.VECTOR_SIZE,
            min_count=cls.MIN_COUNT,
            workers=1,
            alpha=0.002,
            epochs=cls.EPOCHS,
        )
        store = cls(model, path)
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(path))
        return store

    @classmethod
    def load(cls, path: Path) -> EmbeddingStore:
        """Load an existing artifact without a provider or training data."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Word2Vec artifact does not exist: {path}")
        if path.stat().st_size == 0:
            raise ValueError(f"Word2Vec artifact is empty: {path}")
        return cls(Word2Vec.load(str(path)), path)

    def get_vector(self, vocabulary: dict[str, int]) -> torch.Tensor:
        for required_token in ("<PAD>", "<UNK>"):
            if required_token not in vocabulary:
                raise ValueError(
                    f"Vocabulary is missing required token {required_token}"
                )

        keyed_vectors = self.word2vec.wv
        embedding_matrix = torch.empty(
            len(vocabulary), keyed_vectors.vector_size, dtype=torch.float
        )
        torch.nn.init.normal_(embedding_matrix, mean=0.0, std=0.02)
        embedding_matrix[vocabulary["<PAD>"]].zero_()

        word_vectors = torch.from_numpy(keyed_vectors.vectors)
        embedding_matrix[vocabulary["<UNK>"]] = word_vectors.mean(dim=0)

        for token, token_id in vocabulary.items():
            if token in keyed_vectors.key_to_index:
                embedding_matrix[token_id] = torch.from_numpy(
                    keyed_vectors[token].copy()
                )
        return embedding_matrix

