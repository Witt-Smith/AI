"""Checkpoint-compatible GRU training and validation computation."""

import math
import logging
from typing import cast

import lightning as L
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from .tokenizer import Tokenizer

LOGGER = logging.getLogger(__name__)


class Train(L.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        tokenizer: Tokenizer,
        learning_rate: float = 0.002,
    ):
        super().__init__()
        tokenizer_vocabulary_size = len(tokenizer.token_to_id)
        if vocab_size != tokenizer_vocabulary_size:
            raise ValueError(
                "vocab_size must match the tokenizer vocabulary: "
                f"model={vocab_size}, tokenizer={tokenizer_vocabulary_size}"
            )
        if pad_id < 0 or pad_id >= vocab_size:
            raise ValueError("pad_id must be inside the vocabulary")
        if (
            isinstance(learning_rate, bool)
            or not isinstance(learning_rate, (int, float))
            or not math.isfinite(learning_rate)
            or learning_rate <= 0
        ):
            raise ValueError("learning_rate must be a finite positive number")

        self.save_hyperparameters(ignore=["tokenizer"])
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)

        embedding_weights = tokenizer.get_vector(tokenizer.token_to_id)
        embedding_size = embedding_weights.size(1)
        self.embedding = nn.Embedding.from_pretrained(
            embedding_weights,
            freeze=True,
            padding_idx=pad_id,
        )
        self.linear = nn.Linear(
            in_features=embedding_size,
            out_features=vocab_size,
            bias=True,
        )
        self.encoder = nn.GRU(
            input_size=embedding_size,
            hidden_size=embedding_size,
            batch_first=True,
        )
        self.decoder = nn.GRU(
            input_size=embedding_size,
            hidden_size=embedding_size,
            batch_first=True,
        )

    def forward(
        self,
        question_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        question_lengths: torch.Tensor,
    ) -> torch.Tensor:
        question_embedding = self.embedding(question_ids)
        decoder_embedding = self.embedding(decoder_input_ids)

        packed_question = pack_padded_sequence(
            question_embedding,
            question_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, encoder_hidden = self.encoder(packed_question)
        decoder_output, _ = self.decoder(decoder_embedding, encoder_hidden)
        return self.linear(decoder_output)

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        loss = self._sequence_loss(batch)
        question_ids = batch[0]
        optimizer = cast(
            torch.optim.Optimizer,
            self.optimizers(use_pl_optimizer=False),
        )
        learning_rate = optimizer.param_groups[0]["lr"]

        self.log_dict(
            {
                "normal_loss": loss,
                "lr": learning_rate,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=question_ids.size(0),
        )
        return loss

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        loss = self._sequence_loss(batch)
        self.log(
            "validation_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch[0].size(0),
        )
        return loss

    def _sequence_loss(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        question_ids, answer_ids, question_lengths = batch
        decoder_input_ids = answer_ids[:, :-1]
        labels = answer_ids[:, 1:]

        logits = self(question_ids, decoder_input_ids, question_lengths)
        return self.loss_fn(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            params=self.parameters(),
            lr=self.learning_rate,
        )

    def on_train_epoch_end(self) -> None:
        LOGGER.info("Training epoch finished: epoch=%d global_step=%d", self.current_epoch + 1, self.global_step)
