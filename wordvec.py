from collections import Counter
from pathlib import Path
from typing import Optional

import torch
from gensim.models import Word2Vec

from data_provider import DataProvider


class WordVec:
    """Load or train the Word2Vec artifact used to initialise embeddings."""

    VECTOR_SIZE = 128
    MIN_COUNT = 2
    EPOCHS = 1000
    def __init__(
        self,
        data_provider: DataProvider,
        model_path: Optional[Path] = None,
    ):
        self.data_provider = data_provider
        self.model_path = model_path
        self.word2vec = self._load_or_train_model()

        if self.word2vec.vector_size != self.VECTOR_SIZE:
            raise ValueError(
                "Word2Vec vector size does not match the model embedding size: "
                f"expected {self.VECTOR_SIZE}, got {self.word2vec.vector_size}"
            )
        if not self.word2vec.wv.key_to_index:
            raise ValueError("Word2Vec vocabulary is empty")

    def _load_or_train_model(self) -> Word2Vec:
        if self.model_path is not None and self.model_path.is_file():
            return Word2Vec.load(str(self.model_path))

        sentences = self.data_provider.load_word_data()
        token_counts = Counter(token for sentence in sentences for token in sentence)
        if not any(count >= self.MIN_COUNT for count in token_counts.values()):
            raise ValueError(
                "Word2Vec cannot build a vocabulary: no token appears at least "
                f"{self.MIN_COUNT} times"
            )

        model = Word2Vec(
            sentences,
            vector_size=self.VECTOR_SIZE,
            min_count=self.MIN_COUNT,
            workers=1,
            alpha=0.002,
            epochs=self.EPOCHS,
        )
        if self.model_path is not None:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(self.model_path))
        return model

    def train(self, train_file: str, epochs: int = EPOCHS) -> None:
        if epochs <= 0:
            raise ValueError("epochs must be greater than 0")
        self.word2vec.train(
            corpus_file=train_file,
            total_words=self.word2vec.corpus_total_words,
            epochs=epochs,
        )
        if self.model_path is not None:
            self.word2vec.save(str(self.model_path))

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

    def _require_known_words(self, words: tuple[str, ...]) -> None:
        if not words:
            raise ValueError("At least one word is required")
        missing_words = [word for word in words if word not in self.word2vec.wv]
        if missing_words:
            raise KeyError(
                f"Words are not in the Word2Vec vocabulary: {missing_words}"
            )

    def semantic_map(self, *words: str) -> None:
        self._require_known_words(words)

        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA

        keyed_vectors = self.word2vec.wv
        all_points = PCA(n_components=2).fit_transform(
            keyed_vectors.get_normed_vectors()
        )
        word_points = [keyed_vectors.key_to_index[word] for word in words] # type: ignore

        plt.figure(figsize=(12.0, 9.0))
        plt.scatter(
            x=all_points[:, 0],
            y=all_points[:, 1],
            s=23,
            alpha=0.01,
            color="black",
        )
        plt.scatter(
            x=all_points[word_points, 0],
            y=all_points[word_points, 1],
            s=25,
            alpha=1,
            color="red",
        )
        plt.show()

    def most_similar(self, word: str, topn: int = 5) -> list[tuple[str, float]]:
        if topn <= 0:
            raise ValueError("topn must be greater than 0")
        return self.word2vec.wv.most_similar(word, topn=topn)

    def relationship_map(self, *words: str) -> None:
        self._require_known_words(words)

        import matplotlib.pyplot as plt
        import numpy as np

        keyed_vectors = self.word2vec.wv
        matrix = np.array(
            [
                [
                    keyed_vectors.similarity(word1, word2)
                    for word2 in words
                ]
                for word1 in words
            ]
        )

        plt.rcParams["font.family"] = "Arial Unicode MS"
        plt.rcParams["axes.unicode_minus"] = False
        figure, axis = plt.subplots(figsize=(9, 7))
        image = axis.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
        axis.set_xticks(range(len(words)), labels=words, rotation=45)
        axis.set_yticks(range(len(words)), labels=words)
        axis.set_xlabel("对比词 word2")
        axis.set_ylabel("基准词 word1")
        axis.set_title("词语关系程度")

        for row, column in np.ndindex(matrix.shape):
            axis.text(
                column,
                row,
                f"{words[row]} ↔ {words[column]}\n{matrix[row, column]:.2f}",
                ha="center",
                va="center",
            )

        figure.colorbar(image, ax=axis, label="余弦相似度")
        figure.tight_layout()
        plt.show()
