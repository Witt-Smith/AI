from collections.abc import Sequence
from datasets import DatasetDict
from torch import Tensor, long, tensor
import tiktoken

class Tokenizer:
    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    UNK_ID = 3
    SPECIAL_TOKEN_COUNT = 4

    def __init__(self) -> None:
        super().__init__()
        self.token_tale: dict[str, int] = {}
        self.ens = tiktoken.get_encoding("o200k_base")

    def __len__(self) -> int:
        return self.SPECIAL_TOKEN_COUNT + len(self.token_tale)

    def assign_id(self, data: DatasetDict) -> None:
        token_tale: dict[str, int] = {}
        for split in data.values():
            for sample in split:
                for field in ("feature", "label"):
                    text = sample[field]
                    texts = self.cut(text)

                    if not isinstance(text, (str, Sequence)):
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



    def encode(self, text: Sequence[str]) -> Tensor:
        if not self.token_tale:
            raise ValueError("Token table is empty.")

        ids = [self.BOS_ID]
        ids.extend(self.token_tale.get(token, self.UNK_ID) for token in text)
        ids.append(self.EOS_ID)

        return tensor(ids, dtype=long)

    def decode(self, text: list[str]):
        pass
        