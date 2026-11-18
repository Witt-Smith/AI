import torch
import torch.nn as nn
import lightning.pytorch as pl
from torch.optim import AdamW
from Config.ai_config import config
from .transformerBlock import TranormerBlock



class NeuralNetwork(pl.LightningModule):
    def __init__(self, vocab_size: int, intent_names=("identity", "ability", "unknown"),
                 max_seq_len: int = 512, vocabulary=None) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.intent_names = tuple(intent_names)
        self.train_cnf = config.train_cnf
        self.tokenization_cnf = config.tokenization_cnf
        self.vocab_size = vocab_size
        self.pad = int(self.tokenization_cnf["pad_idx"])
        self.eos = int(self.tokenization_cnf["eos_idx"])
        self.sep = int(self.tokenization_cnf["sep_idx"])
        self.eps = self.train_cnf['eps']
        self.embedding_dim = self.train_cnf['embedding_dim']
        self.hidden_size = self.train_cnf['hidden_size']
        self.max_seq_len = max_seq_len

        if not 0 <= self.pad < vocab_size:
            raise ValueError(f"pad_idx {self.pad} is outside the vocabulary.")

        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=self.embedding_dim,
            padding_idx=self.pad,
        )
        self.position_embedding = nn.Embedding(
            num_embeddings=self.max_seq_len,
            embedding_dim=self.embedding_dim,
        )
        self.register_buffer(
            "causal_mask", torch.ones(max_seq_len, max_seq_len, dtype=torch.bool).triu(1),
            persistent=False,
        )
        # 保留旧检查点中的参数，实际生成使用 lmh。
        self.output_layer = nn.Linear(self.hidden_size * 2, vocab_size).requires_grad_(False)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad)
        self.intent_loss_fn = nn.CrossEntropyLoss()
        self.transformer_block = nn.ModuleList([
            TranormerBlock(self.embedding_dim, 4, self.eps)
            for _ in range(5)
        ])
        self.lmh = nn.Linear(self.embedding_dim,vocab_size)
        self.intent_head = nn.Linear(
            self.embedding_dim,
            len(self.intent_names)
        )
                
    def forward(self, features):
        if features.size(1) > self.max_seq_len:
            raise ValueError(f"输入长度 {features.size(1)} 超过模型上限 {self.max_seq_len}")
        position_features = torch.arange(features.size(1), device=features.device)
        input_embedding = self.token_embedding(features) + self.position_embedding(position_features)

        pad_mask = features.eq(self.pad)
        sep_mask = features.eq(self.sep)
        sep_positions = sep_mask.long().argmax(dim=1)

        causal_mask = self.causal_mask[:features.size(1), :features.size(1)]
        matrix = input_embedding
        for block in self.transformer_block:
            matrix = block(matrix, pad_mask, causal_mask)
        logits = self.lmh(matrix)

        # SEP 的隐藏状态能看到整个问题，训练和聊天时都看不到回答。
        question = matrix[torch.arange(features.size(0), device=features.device), sep_positions]
        intent = self.intent_head(question)
        return logits, intent

    def _shared_step(self, batch, stage):
        features, labels, intent_labels = batch
        logits,intent = self(features)
        loss = self.loss_fn(logits.reshape(-1, self.vocab_size), labels.reshape(-1))
        if stage == "train":
            intent_loss = self.intent_loss_fn(intent, intent_labels)
            total_loss = loss + intent_loss
            self.log("train_loss", loss, on_step=True, on_epoch=True)
            self.log("intent_loss", intent_loss, on_step=True, on_epoch=True)
            self.log("total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
            return total_loss

        self.log(f"{stage}_reply_loss", loss, batch_size=int(labels.ne(self.pad).sum()))
        self.log(f"{stage}_intent_acc", (intent.argmax(1) == intent_labels).float().mean(),
                 batch_size=len(intent_labels))

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def configure_optimizers(self):

        return AdamW(
            self.parameters(),
            lr=3e-4,
            weight_decay=0.0,
        )
