import torch.nn as nn
from torch import Tensor
from math import sqrt
from torch import softmax

class MultiHeadAttention():

    def __init__(self,embed_dim: int,num_heads: int,batch_first:bool = True) -> None:
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.heads_dim = embed_dim // num_heads
        self.batch_first = batch_first
        self.Wq = nn.Linear(self.embed_dim , self.embed_dim )
        self.Wk = nn.Linear(self.embed_dim , self.embed_dim )
        self.Wv = nn.Linear(self.embed_dim , self.embed_dim )





    def compute_attention_scores(self,matrix: Tensor)-> Tensor:
        global B,T,D

        B,T,D = matrix.shape
        q = self.Wq(matrix).view(B, T, self.num_heads, self.heads_dim).transpose(1, 2)
        k = self.Wk(matrix).view(B, T, self.num_heads, self.heads_dim).transpose(1, 2)
        v = self.Wv(matrix).view(B, T, self.num_heads, self.heads_dim).transpose(1, 2)

        scores = q @ k.transpose(-2, -1)
        scores = scores / sqrt(self.heads_dim)
        weights = softmax(scores, dim=-1)

        return weights @ v 


    def forward(self,matrix: Tensor)-> Tensor:
        weights = self.compute_attention_scores(matrix)
        weights = weights.transpose(1,2).contiguous().view(B,T,D)

        return weights


    