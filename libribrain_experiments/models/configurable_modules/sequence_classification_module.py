import torch
from torch import nn
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy, F1Score, Precision, Recall

from .utils import loss_fn_from_config, modules_from_config, optimizer_from_config


class SequenceClassificationModule(LightningModule):
    """
    Version of ClassificationModule that expects targets with shape (B,T)
    and models producing logits of shape (B*T, n_classes).
    """

    def __init__(
        self, model_config: list[tuple[str, dict]], n_classes: int,
        optimizer_config: dict, loss_config: dict,
        single_logit: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.single_logit = single_logit
        self.modules_list = nn.ModuleList()
        self.modules_list.extend(modules_from_config(model_config))
        self.loss_fn = loss_fn_from_config(loss_config)
        if self.single_logit:  # 1-logit + sigmoid
            n_classes = 2
        self.accuracy = Accuracy(num_classes=n_classes, task="multiclass")
        self.balanced_accuracy = Accuracy(
            num_classes=n_classes, task="multiclass", average="macro")
        self.f1_micro = F1Score(num_classes=n_classes,
                                task="multiclass", average="micro")
        self.f1_macro = F1Score(num_classes=n_classes,
                                task="multiclass", average="macro")
        self.precision_micro = Precision(
            num_classes=n_classes, task="multiclass", average="micro")
        self.precision_macro = Precision(
            num_classes=n_classes, task="multiclass", average="macro")

        self.optimizer_config = optimizer_config

        self.binary_accuracy = Accuracy(task="binary")
        self.binary_precision = Precision(task="binary")
        self.binary_recall = Recall(task="binary")
        self.binary_f1 = F1Score(task="binary")

    def forward(self, x):
        for m in self.modules_list:
            x = m(x)
        return x  # (B*T, n_classes)

    def configure_optimizers(self):
        return optimizer_from_config(self.parameters(), self.optimizer_config)

    def _shared_step(self, batch, stage: str):
        x, y = batch  # x:(B,C,T), y:(B,T)
        B, T = y.shape
        y = y.reshape(-1)  # (B*T,)
        y_hat = self(x)  # (B*T,n_classes)
        loss = self.loss_fn(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)

        acc = self.accuracy(preds, y)
        f1_micro = self.f1_micro(preds, y)
        f1_macro = self.f1_macro(preds, y)
        bal_acc  = self.balanced_accuracy(preds, y)
        self.log(f'{stage}_loss', loss)
        self.log(f'{stage}_acc', acc)
        self.log(f'{stage}_f1_micro', f1_micro)
        self.log(f'{stage}_f1_macro', f1_macro)
        self.log(f'{stage}_precision_micro', self.precision_micro(preds, y))
        self.log(f'{stage}_precision_macro', self.precision_macro(preds, y))
        self.log(f'{stage}_bal_acc', bal_acc)
        for class_idx in range(y_hat.shape[1]):
            y_binary = (y == class_idx).int()
            y_hat_binary = y_hat.argmax(dim=1) == class_idx
            binary_acc = self.binary_accuracy(y_hat_binary, y_binary)
            self.log(f'{stage}_acc_class_{class_idx}', binary_acc)
            monitored_class_prediction_rate = y_hat_binary.float().mean()
            self.log(f'{stage}_prediction_rate_class_{class_idx}',
                     monitored_class_prediction_rate)
            monitored_class_recall = self.binary_recall(y_hat_binary, y_binary)
            self.log(f'{stage}_recall_class_{class_idx}', monitored_class_recall)
            monitored_class_precision = self.binary_precision(
                y_hat_binary, y_binary)
            self.log(f'{stage}_precision_class_{class_idx}',
                     monitored_class_precision)
            y_hat_probs = nn.functional.softmax(y_hat, dim=1)
            monitored_class_mean_probs = y_hat_probs[:, class_idx].mean(
            )
            self.log(f'{stage}_mean_probability_class_{class_idx}',
                     monitored_class_mean_probs)

            self.log(f'{stage}_f1_class_{class_idx}',
                     self.binary_f1(y_hat_binary, y_binary))
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")
