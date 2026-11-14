import torch 
from torch.nn.parameter import Parameter
from torch import long,Tensor

class PositionEmbedding():

    def __init__(self,num_embeddings: int,embedding_dim: int) -> None:
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self._weight = Parameter(
            data = torch.empty(num_embeddings,embedding_dim,dtype = long ),
            requires_grad = True
        )



    def pe(self,pos: Tensor,i: Tensor)-> Tensor:

        return pos / pow(10000,2*i/self.embedding_dim)

