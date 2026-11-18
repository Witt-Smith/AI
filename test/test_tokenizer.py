from Tokenizer.tokenizer import Tokenizer
from json import load
from pathlib import Path
from Model.MultiHeadAttention import MultiHeadAttention
from torch import float
from torch import arange
from torch import isfinite


tokenizer = Tokenizer()
mlt_head_aten = MultiHeadAttention(144,12)


def test_participle():
    tokenizer = Tokenizer()
    path = Path(__file__).parents[1] / "TrainDataSet/test.json"
    with path.open(encoding="utf-8") as f:
        data = load(f)["data"]
    tokenizer.assign_ids(data)
    sample = data["train"][0]
    encoded = tokenizer.encode(sample["feature"], sample["label"]).tolist()
    sep = encoded.index(tokenizer.sep)
    assert tokenizer.decode(encoded[1:sep]) == sample["feature"]
    assert tokenizer.decode(encoded[sep + 1:-1]) == sample["label"]


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


def test_future_tokens_do_not_change_prefix_or_intent():
    import torch
    from Model.network import NeuralNetwork

    model = NeuralNetwork(16).eval()
    sequences = torch.tensor([[2, 6, 3, 7, 8], [2, 6, 3, 9, 0]])
    with torch.inference_mode():
        logits, intents = model(sequences)
        torch.testing.assert_close(logits[0, :3], logits[1, :3])
        torch.testing.assert_close(intents[0], intents[1])


def test_question_groups_and_answer_targets():
    from Data.dataset import DataSet

    tokenizer = Tokenizer()
    dataset = DataSet(tokenizer)
    splits = [dataset.train_dataset, dataset.valid_dataset, dataset.test_dataset]
    train, valid, test = [{r["feature"] for r in split.samples} for split in splits]
    assert train and valid and test
    assert train.isdisjoint(valid | test) and valid.isdisjoint(test)
    tokenizer.assign_ids({"train": splits[0].samples})
    sequence, category = splits[1][0]
    features, targets, labels = dataset.collate_batch([(sequence, category)])
    sep = sequence.tolist().index(tokenizer.sep)
    assert features[0].tolist() == sequence[:-1].tolist()
    assert targets[0, :sep].eq(tokenizer.pad).all()
    assert targets[0, sep:].tolist() == sequence[sep + 1:].tolist()
    assert labels.item() == dataset.intent_to_id[category]
