from json import load
from pathlib import Path
from random import Random
from torch.utils.data.dataset import Dataset
from Tokenizer.tokenizer import Tokenizer
from datasets import DatasetDict
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from Config.ai_config import config
from torch import long, tensor

class DataSet():
    def __init__(self, tokenizer: Tokenizer) -> None:
        super().__init__()
        self.cnf = config.config
        self.data_cnf = config.data_cnf
        self.tokenization_cnf = config.tokenization_cnf
        self.tokenizer = tokenizer
        self.source = self.data_cnf["source"]
        self.source_path = Path(__file__).parents[1] / Path(self.data_cnf["source_path"]).expanduser()
        self.ds = self.loading_data()
        self.pad = self.tokenization_cnf['pad_idx']
        self.sep = self.tokenization_cnf['sep_idx']
        splits = self.ds
        if self.source == "local" and not (len(splits["validation"]) or len(splits["test"])):
            splits = self.split_questions(splits["train"])
        self.train_dataset = TrainDataset(splits["train"], tokenizer)
        self.valid_dataset = TrainDataset(splits["validation"], tokenizer)
        self.test_dataset = TrainDataset(splits["test"], tokenizer)
        categories = sorted({row["category"] for row in self.train_dataset.samples})
        self.intent_to_id = {name: i for i, name in enumerate(categories)}

    @staticmethod
    def split_questions(rows):
        # 同一问题的所有回答进入同一个集合，每个类别至少保留一个训练问题。
        questions = {}
        for row in rows:
            questions.setdefault(row["category"], set()).add(row["feature"])
        rng = Random(42)
        assigned = {}
        for category in sorted(questions):
            group = sorted(questions[category])
            rng.shuffle(group)
            count = min(max(1, len(group) // 10), (len(group) - 1) // 2)
            for index, question in enumerate(group):
                if index < count:
                    split = "validation"
                elif index < 2 * count:
                    split = "test"
                else:
                    split = "train"
                assigned.setdefault(question, split)
        for group in questions.values():
            if not any(assigned[q] == "train" for q in group):
                assigned[min(group)] = "train"
        splits = {"train": [], "validation": [], "test": []}
        for row in rows:
            splits[assigned[row["feature"]]].append(row)
        return splits

    def loading_data(self) -> DatasetDict:
        if self.source == "local":
            if not self.source_path.is_file():
                raise FileNotFoundError(f"Dataset file not found: {self.source_path}")
            if self.source_path.suffix.lower() != ".json":
                raise ValueError(f"Unsupported dataset file; expected .json: {self.source_path}")
            try:
                with open(self.source_path, mode="r", encoding="utf-8") as f:
                    data = load(f)
                    if not isinstance(data, dict):
                        raise TypeError(f"Expected JSON object, got {type(data).__name__}")
                    
                    return self.covert(data)
            except (OSError, ValueError, TypeError) as e:
                raise ValueError(f"Failed to load dataset from {self.source_path}: {e}") from e

        if self.source == "net":
            dataset = load_dataset("silver/lccc", "base")
            if not isinstance(dataset, DatasetDict):
                raise TypeError(f"Expected DatasetDict, got {type(dataset).__name__}")
            
            return dataset

        raise ValueError(f"Unsupported source '{self.source}'. Expected 'local' or 'net'.")

    def covert(self, data: dict) -> DatasetDict:
        if "data" not in data or not isinstance(data["data"], dict):
            raise TypeError("Expected JSON object with a 'data' field.")

        splits = data["data"]
        return DatasetDict({
            "train": splits.get("train", []),
            "validation": splits.get("validation", []),
            "test": splits.get("test", []),
        })
    
    def collate_batch(self, batch):
        sequences, categories = zip(*batch)
        features = pad_sequence([s[:-1] for s in sequences], batch_first=True, padding_value=self.pad)
        labels = pad_sequence([s[1:] for s in sequences], batch_first=True, padding_value=self.pad)
        # SEP 所在位置预测回答的第一个 token，之前的位置不计入回答损失。
        labels.masked_fill_(features.eq(self.sep).cumsum(dim=1) == 0, self.pad)
        intents = tensor([self.intent_to_id.get(name, -1) for name in categories], dtype=long)
        return features, labels, intents


class TrainDataset(Dataset):
    def __init__(self, samples, tokenizer: Tokenizer) -> None:
        self.samples = samples
        self.tokenizer = tokenizer

    def __getitem__(self, index: int):
        sample = self.samples[index]
        return self.tokenizer.encode(sample["feature"], sample["label"]), sample["category"]

    def __len__(self) -> int:
        return len(self.samples)
