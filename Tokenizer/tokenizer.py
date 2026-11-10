from collections.abc import Sequence
from datasets import DatasetDict
from torch import Tensor, long, tensor
import tiktoken
from Config.ai_config import config

class Tokenizer:
    SPECIAL_TOKEN_COUNT = 5

    def __init__(self) -> None:
        super().__init__()
        self.token_tale: dict[str, int] = {}
        self.ens = tiktoken.get_encoding("cl100k_base")
        self.tokenization_cnf = config.tokenization_cnf
        self.pad = self.tokenization_cnf['pad_idx']
        self.eos = self.tokenization_cnf['eos_idx']
        self.bos = self.tokenization_cnf['bos_idx']
        self.sep = self.tokenization_cnf['sep_idx']
        self.unk = self.tokenization_cnf['unk_idx']

    def __len__(self) -> int:
        ''' Return the vocabluary size. '''

        return self.SPECIAL_TOKEN_COUNT + len(self.token_tale)

    def assign_id(self, data: DatasetDict) -> None:
        ''' Assign Token id. '''

        token_tale: dict[str, int] = {}
        for split in data.values():
            for sample in split:
                for field in ("feature", "label"):
                    text = sample[field] # type: ignore

                    if isinstance(text, str):
                        texts = self.cut(text)
                    elif isinstance(text, Sequence):
                        if not all(isinstance(token, str) for token in text):
                            raise TypeError(
                                f"Expected '{field}' tokens to be strings."
                            )
                        texts = text
                    else:
                        raise TypeError(
                            f"Expected '{field}' to be text or a sequence of tokens, "
                            f"got {type(text).__name__}"
                        )
                    for token in texts:
                        if token not in token_tale:
                            token_tale[token] = self.SPECIAL_TOKEN_COUNT + len(token_tale)

        self.token_tale = token_tale

    def cut(self,text: str)-> list[str]:
        ids = self.ens.encode(text)
        parts = [self.ens.decode_single_token_bytes(i) for i in ids]
        result = list()
        buffer = b''
        for token in parts:
            buffer += token
            try:
                result.append(buffer.decode('utf-8'))
                buffer = b''
            except UnicodeDecodeError as e:
                raise ValueError(f"Unicode Decoder Error {e}")
        else:
            return result

    def encode(self,*sequence) -> Tensor:
        if not self.token_tale:
            raise ValueError("Token table is empty.")

        ids = [self.bos]
        ids.extend(self.token_tale.get(token, self.unk) for token in sequence[0])
        ids.append(self.sep)
        ids.extend(self.token_tale.get(token, self.unk) for token in sequence[1])
        ids.append(self.eos)
        
        return tensor(ids, dtype=long)

    def decode(self, text: list[str]):
        pass
        