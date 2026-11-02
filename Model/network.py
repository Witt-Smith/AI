import torch
import torch.nn as nn
import lightning.pytorch as pl
from torch.optim import AdamW
from omegaconf import DictConfig, OmegaConf
from pathlib import Path


class NeuralNetwork(pl.LightningModule):
    PATH_YAML = Path(__file__).resolve().parent.parent / "config.yaml"
    EMBEDDING_DIM = 128
    HIDDEN_SIZE = 256

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.cnf = OmegaConf.load(self.PATH_YAML)

        if not isinstance(self.cnf, DictConfig):
            raise TypeError(f"Expected DictConfig, got {type(self.cnf)}")

        self.train_cnf = self.cnf["train"]
        self.vocab_size = vocab_size
        self.pad_idx = int(self.train_cnf["pad_idx"])

        if not 0 <= self.pad_idx < vocab_size:
            raise ValueError(f"pad_idx {self.pad_idx} is outside the vocabulary.")

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=self.EMBEDDING_DIM,
            padding_idx=self.pad_idx,
        )
        self.encoder = nn.GRU(
            input_size=self.EMBEDDING_DIM,
            hidden_size=self.HIDDEN_SIZE,
            batch_first=True,
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=self.HIDDEN_SIZE,
            num_heads=4,
            batch_first=True,
        )
        self.decoder = nn.GRU(
            input_size=self.EMBEDDING_DIM,
            hidden_size=self.HIDDEN_SIZE,
            batch_first=True,
        )
        self.output_layer = nn.Linear(self.HIDDEN_SIZE * 2, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_idx)

    def forward(self, features, labels):
        feature_lengths = (features != self.pad_idx).sum(dim=1).cpu()
        embedded_features = self.embedding(features)
        packed_features = nn.utils.rnn.pack_padded_sequence(
            embedded_features,
            feature_lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_encoder_output, encoder_state = self.encoder(packed_features)
        encoder_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_encoder_output,
            batch_first=True,
            total_length=features.size(1),
        )

        decoder_inputs = self.embedding(labels[:, :-1])
        decoder_output, _ = self.decoder(decoder_inputs, encoder_state)
        attention_output, _ = self.attention(
            query=decoder_output,
            key=encoder_output,
            value=encoder_output,
            key_padding_mask=features == self.pad_idx,
            need_weights=False,
        )
        logits = self.output_layer(
            torch.cat((decoder_output, attention_output), dim=-1)
        )

        return logits

    def training_step(self, batch, batch_idx):
        features, labels = batch
        logits = self(features, labels)
        targets = labels[:, 1:]
        loss = self.loss_fn(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return AdamW(
            self.parameters(),
            lr=3e-4,
            weight_decay=0.0,
        )
