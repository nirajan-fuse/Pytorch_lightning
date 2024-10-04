from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback


class MyPrintingCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_train_start(self, trainer, pl_module):
        print("starting to train!")

    def on_train_end(self, trainer, pl_module):
        print("training is done.")
