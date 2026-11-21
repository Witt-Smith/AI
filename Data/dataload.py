from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from Data.dataset import DataSet


class DataLoad(DataSet):
    def __init__(self, tokenizer, *, source_path: Path | None = None,
                 batch_size: int | None = None, num_workers: int | None = None):
        super().__init__(tokenizer, source_path=source_path)
        self.batch_size = self.data_cnf["batch_size"] if batch_size is None else batch_size
        self.shuffle = self.data_cnf["shuffle"]
        self.num_workers = self.data_cnf["num_workers"] if num_workers is None else num_workers
        self.timeout = self.data_cnf["timeout"]
        self.pin_memory = self.data_cnf["pin_memory"]

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            timeout=self.timeout,
            collate_fn=self.collate_batch,
        )
        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            timeout=self.timeout,
            collate_fn=self.collate_batch,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            timeout=self.timeout,
            collate_fn=self.collate_batch,
        )  
    
