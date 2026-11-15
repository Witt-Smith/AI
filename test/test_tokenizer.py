from Tokenizer.tokenizer import Tokenizer
from datasets import DatasetDict
from json import load
from pathlib import Path
from Tokenizer.tokenizer import Tokenizer
from Model.MultiHeadAttention import MultiHeadAttention
from torch import float
from torch import arange
from torch import isfinite


tokenizer = Tokenizer()
mlt_head_aten = MultiHeadAttention(144,12)


def test_participle():
    PATH = Path(__file__).parents[1] / 'TrainDataSet/test.json'
    with open(PATH,mode="r",encoding="utf-8") as f:
        splits = load(f)
        if not isinstance(splits, dict):
            raise TypeError(f"Expected JSON object, got {type(splits).__name__}")
        data = DatasetDict({
            "train": splits.get("train", []),
            "validation": splits.get("validation", []),
            "test": splits.get("test", []),
        })
    tokenizer.assign_ids(data)

    assert len(tokenizer.token_tale) == 0,'Writen Errror!!!'


def test_cut_handles_multibyte_utf8_tokens():
    text = "你好，世界！🙂"

    tokens = tokenizer.cut(text)

    assert tokens
    assert "".join(tokens) == text


def test_forward():
    matrix = arange(3600,dtype=float).reshape(5,5,144)
    output = mlt_head_aten.forward(matrix)

    assert output.shape == matrix.shape
    assert isfinite(output).all()
