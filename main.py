import lightning.pytorch as pl
from Tokenizer.tokenizer import Tokenizer
from Data.dataload import DataLoad
from Model.network import NeuralNetwork





if __name__ == "__main__":
    
    tokenizer = Tokenizer()
    dataloader = DataLoad(tokenizer)
    tokenizer.assign_id(dataloader.ds) # type: ignore
    network = NeuralNetwork(len(tokenizer))
    trainer = pl.Trainer(
        max_epochs = 100,
        accelerator='auto',
        devices='auto'
    )
    trainer.fit(
        network,
        train_dataloaders=dataloader.train_loader
    )
    