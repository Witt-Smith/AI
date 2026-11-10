
import torch.nn as nn


class TranormerBlock(nn.Module):

    
    def __init__(self, embedding_dim: int, num_heads: int, eps: float) -> None:
        super().__init__()
        self.normal_layer = nn.LayerNorm(
            normalized_shape=embedding_dim,
            eps=eps,
            elementwise_affine=True,
            bias=True,
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, input_embedding, padding_mask=None):
        resdual = input_embedding
        normalized_output = self.normal_layer(input_embedding)
        attention_output, attention_weights = self.attention(
            query=normalized_output,
            key=normalized_output,
            value=normalized_output,
            key_padding_mask=padding_mask,
            need_weights=False,
        )

        return self.network(attention_output + resdual)