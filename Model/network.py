import torch
import torch.nn as nn
import lightning.pytorch as pl
from torch.optim import AdamW
from Config.ai_config import config
from .transformerBlock import TranormerBlock
from torch import Tensor



class NeuralNetwork(pl.LightningModule):
    intent_to_id = {
        "identity": 0,   # 询问身份
        "ability": 1,    # 询问能力
        "unknown": 2,    # 数据集范围以外
    }
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.train_cnf = config.train_cnf
        self.tokenization_cnf = config.tokenization_cnf
        self.vocab_size = vocab_size
        self.pad = int(self.tokenization_cnf["pad_idx"])
        self.eos = int(self.tokenization_cnf["eos_idx"])
        self.sep = int(self.tokenization_cnf["sep_idx"])
        self.eps = self.train_cnf['eps']
        self.embedding_dim = self.train_cnf['embedding_dim']
        self.hidden_size = self.train_cnf['hidden_size']
        self.max_seq_len = 512

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
            padding_idx=self.pad,
        )
        self.output_layer = nn.Linear(self.hidden_size * 2, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad)
        self.intent_loss_fn = nn.CrossEntropyLoss()
        self.transformer_block = nn.ModuleList([
            TranormerBlock(self.embedding_dim, 4, self.eps)
            for _ in range(50)
        ])
        self.lmh = nn.Linear(self.embedding_dim,vocab_size)
        self.intent_head = nn.Linear(
            self.embedding_dim,
            len(self.intent_to_id)
        )
                
    def forward(self, features, labels):
        position_features = torch.arange(features.size(1), device=features.device)
        input_embedding = self.token_embedding(features) + self.position_embedding(position_features)

        pad_mask = features.eq(self.pad)
        eosing_mask = features.eq(self.eos)
        sep_mask = features.eq(self.sep)
        sep_positions = sep_mask.long().argmax(dim=1)
        matrix = input_embedding
        for block in self.transformer_block:
            matrix = block(matrix, pad_mask)
        logits = self.lmh(matrix)
        valid_mask = features.ne(self.pad)
        pooled = self.polling(matrix, valid_mask)
        intent = self.intent_head(pooled)

        return logits, intent

    
    

        
    def polling(self, data: Tensor, valid_mask: Tensor) -> Tensor:
        mask = valid_mask.unsqueeze(-1).to(data.dtype)
        summed = (data * mask).sum(dim=1)       
        lengths = mask.sum(dim=1).clamp_min(1)  

        return summed / lengths               


    def training_step(self, batch, batch_idx):
        features, labels, intent_labels = batch
        logits,intent = self(features, labels)
        targets = labels[:, 1:]
        loss = self.loss_fn(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
        intent_loss = self.intent_loss_fn(intent,intent_labels)
        total_loss = loss + intent_loss 

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("intent_loss", intent_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)

        return total_loss

    def configure_optimizers(self):

        return AdamW(
            self.parameters(),
            lr=3e-4,
            weight_decay=0.0,
        )
    