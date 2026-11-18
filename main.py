import lightning.pytorch as pl
from Tokenizer.tokenizer import Tokenizer
from Data.dataload import DataLoad
from Model.network import NeuralNetwork

if __name__ == "__main__":

    tokenizer = Tokenizer()
    dataloader = DataLoad(tokenizer)
    vocabulary = tokenizer.assign_ids({"train": dataloader.train_dataset.samples})
    network = NeuralNetwork(len(tokenizer), intent_names=list(dataloader.intent_to_id),
                            vocabulary=vocabulary)
    trainer = pl.Trainer(
        max_epochs = 50,
        accelerator='auto',
        devices='auto'
    )
    trainer.fit(
        network,
        train_dataloaders=dataloader.train_loader,
        val_dataloaders=dataloader.valid_loader if len(dataloader.valid_dataset) else None,
    )
    if len(dataloader.test_dataset):
        trainer.test(network, dataloaders=dataloader.test_loader)
