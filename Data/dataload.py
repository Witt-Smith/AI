from torch.utils.data import DataLoader
from omegaconf import DictConfig
from Data.dataset import DataSet


class DataLoad(DataSet):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        if not isinstance(self.cnf, DictConfig):
            raise TypeError(f"Expected DictConfig, got {type(self.cnf)}")

        self.data_cnf = self.cnf["data"]
        self.batch_size = self.data_cnf["batch_size"]
        self.shuffle = self.data_cnf["shuffle"]
        self.num_workers = self.data_cnf["num_workes"]
        self.timeout = self.data_cnf["timeout"]
        self.pin_memory = self.data_cnf["pin_menory"]

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
    
