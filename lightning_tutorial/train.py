import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import PyTorchProfiler

from model import NN
from dataset import MnistDataModule
import config
from callbacks import MyPrintingCallback

if __name__ == "__main__":
    logger = TensorBoardLogger("tb_logs", name="mnist_model_v1")
    profiler = PyTorchProfiler(
        schedule=torch.profiler.schedule(skip_first=10, wait=1, warmup=1, active=20),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./tb_logs/profiler0"),
    )
    model = NN(
        input_size=config.INPUT_SIZE,
        learning_rate=config.LEARNING_RATE,
        num_classes=config.NUM_CLASSES,
    )
    dm = MnistDataModule(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )
    trainer = L.Trainer(
        strategy='ddp',
        profiler=profiler,
        logger=logger,
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        min_epochs=config.MIN_EPOCHS,
        max_epochs=config.MAX_EPOCHS,
        precision=config.PRECISION,
        callbacks=[MyPrintingCallback(), EarlyStopping(monitor="val_loss")],
    )
    trainer.fit(model, dm)
    trainer.validate(model, dm)
    trainer.test(model, dm)
