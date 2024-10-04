import torch
import torch.nn.functional as F
import torchvision
import torchmetrics.aggregation
from torch import nn
import lightning as L
import torchmetrics


class NN(L.LightningModule):
    def __init__(self, input_size, learning_rate, num_classes):
        super().__init__()
        self.learning_rate = learning_rate
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        X, y = batch
        loss, score, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(score, y)
        f1_score = self.f1_score(score, y)
        self.log_dict(
            {
                "train_loss": loss,
                "train_accuracy": accuracy,
                "train_f1_score": f1_score,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        if batch_idx % 100 == 0:
            X = X[:8]
            grid = torchvision.utils.make_grid(X.view(-1, 1, 28, 28))
            self.logger.experiment.add_image("mnist_images", grid, self.global_step)

        return {"loss": loss, "score": score, "y": y}
    

    def validation_step(self, batch, batch_idx):
        loss, score, y = self._common_step(batch, batch_idx)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss, score, y = self._common_step(batch, batch_idx)
        self.log("test_loss", loss)
        return loss

    def _common_step(self, batch, batch_idx):
        X, y = batch
        X = X.reshape(X.size(0), -1)
        score = self.forward(X)
        loss = self.loss_fn(score, y)
        return loss, score, y

    def prediction_step(self, batch, batch_idx):
        X, y = batch
        X = X.reshape(X.size(0), -1)
        score = self.forward(X)
        pred = torch.argmax(score, dim=1)
        return pred

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
