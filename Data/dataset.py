from json import load
from pathlib import Path
from torch.utils.data.dataset import Dataset
from Tokenizer.tokenizer import Tokenizer
from datasets import DatasetDict
from datasets import load_dataset
from Config.ai_config import config
from torch.nn.utils.rnn import pad_sequence
from datasets import Dataset as HFDataset

class DataSet():
    
    def __init__(self, tokenizer: Tokenizer) -> None:
        super().__init__()
        self.data_cnf = config.data_cnf
        self.tokenization_cnf = config.tokenization_cnf
        self.tokenizer = tokenizer
        self.source = self.data_cnf["source"]
        self.source_path = Path(self.data_cnf["source_path"]).expanduser()
        self.ds = self.loading_data()
        self.pad = self.tokenization_cnf['pad_idx']
        self.eos = self.tokenization_cnf['eos_idx']
        self.bos = self.tokenization_cnf['bos_idx']

        self.train_dataset = TrainDataset(self.ds['train'], self.tokenizer,self.pad,self.eos,self.bos) 
        self.valid_dataset = ValidDataset(self.ds['validation'], self.tokenizer,self.pad,self.eos,self.bos)
        self.test_dataset = TestDataset(self.ds['test'], self.tokenizer,self.pad,self.eos,self.bos) 

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
        features, labels, categories = zip(*batch)
        features = pad_sequence(
            list(features), batch_first=True, padding_value=self.pad
        )
        labels = pad_sequence(
            list(labels), batch_first=True, padding_value=self.pad
        )

        return features, labels, categories


class TrainDataset(Dataset):

    def __init__(self, ds: HFDataset, tokenizer: Tokenizer,pad_idx: int,eos_idx: int,bos_idx: int) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.samples = ds
        self.pad = pad_idx
        self.eos = eos_idx
        self.bos = bos_idx

    def __getitem__(self, index: int):
        sample = self.samples[index]
        sequence = self.tokenizer.encode(sample["feature"],sample["label"])

        return sequence, sample["category"]

    def __len__(self) -> int:

        return len(self.samples)



class ValidDataset(Dataset):

    def __init__(self, ds: HFDataset, tokenizer: Tokenizer,pad_idx: int,eos_idx: int,bos_idx: int) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.samples = ds
        self.bos = bos_idx

    def __getitem__(self, index: int):
        sample = self.samples[index]
        features = self.tokenizer.encode(sample["feature"])
        labels = self.tokenizer.encode(sample["label"])

        return features, labels, sample["category"]

    def __len__(self) -> int:

        return len(self.samples)



class TestDataset(Dataset):

    def __init__(self, ds: HFDataset, tokenizer: Tokenizer,pad_idx: int,eos_idx: int,bos_idx: int) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.samples = ds
        self.bos = bos_idx

    def __getitem__(self, index: int):
        sample = self.samples[index]
        features = self.tokenizer.encode(sample["feature"])
        labels = self.tokenizer.encode(sample["label"])

        return features, labels, sample["category"]

    def __len__(self) -> int:

        return len(self.samples)

    