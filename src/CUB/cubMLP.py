from abc import abstractmethod
import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, matthews_corrcoef
import pytorch_lightning as pl
from torch.nn.functional import binary_cross_entropy, cross_entropy
import torch
from collections import Counter
import torchvision.models as models


class CUBMLP(pl.LightningModule):
    def __init__(self, encoder, embedding_size=2048, num_classes=200):
        super(CUBMLP, self).__init__()
        self.encoder = encoder
        #self.fc1 = torch.nn.Linear(embedding_size, embedding_size)
        #self.fc2 = torch.nn.Linear(embedding_size, embedding_size)
        #self.fc3 = torch.nn.Linear(embedding_size, num_classes)
        self.fc3 = torch.nn.Identity()
        self.loss = torch.nn.CrossEntropyLoss()
    

    def forward(self, x):
        x = self.encoder(x)
        #x = torch.relu(self.fc1(x))
        #x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

    def training_step(self, batch, batch_idx):
        x, c, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y.argmax(dim=-1))
        self.log('train/loss', loss)
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == y.argmax(dim=-1)).float().mean()
        self.log('train/acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, c, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y.argmax(dim=-1))
        self.log('val/loss', loss)
        # accuracy
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == y.argmax(dim=-1)).float().mean()
        self.log('val/acc', acc, prog_bar=True, on_step=False, on_epoch=True)
    

    def test_step(self, batch, batch_idx):
        x, c, y = batch
        logits = self.forward(x)
        # accuracy
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == y.argmax(dim=-1)).float().mean()
        self.log('test/acc', acc)
    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val/loss" 
            },
        }


class fullCUBMLP(pl.LightningModule):
    def __init__(self, nClasses):
        super().__init__()
        self.model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        self.output_dim = nClasses
        self.nClasses = nClasses
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, self.output_dim)
        self.lr = 0.001
        

    def forward(self, batch_x):
        if self.training:
            return self.model(batch_x)[0]
        return self.model(batch_x)

    def training_step(self, batch, batch_idx):
        images, concepts, labels = batch
        outputs = self.forward(images)
        loss = cross_entropy(outputs, labels.argmax(dim=1))
        self.log('train/loss', loss)
        y_pred = torch.nn.functional.one_hot(outputs.argmax(dim=-1), self.nClasses).float()

        with torch.no_grad():
            task_acc = (y_pred == labels).float().mean()
            self.log("train/task_flat_acc", task_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        with torch.no_grad():
            subset_acc = ((y_pred == labels).float().prod(dim=-1)).mean()
            self.log("train/task_subset_acc", subset_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        # Most important!!!
        with torch.no_grad():
            y_true_mc = labels.argmax(dim=-1)
            y_pred_mc = y_pred.argmax(dim=-1)
            multiclass_acc = (y_true_mc == y_pred_mc).float().mean()
            self.log("train/task_multiclass_acc", multiclass_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, concepts, labels = batch
        outputs = self.forward(images)
        loss = cross_entropy(outputs, labels.argmax(dim=1))
        self.log('val/loss', loss)
        y_pred = torch.nn.functional.one_hot(outputs.argmax(dim=-1), self.nClasses).float()
        with torch.no_grad():
            task_acc = (y_pred == labels).float().mean()
            self.log("val/task_flat_acc", task_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        with torch.no_grad():
            subset_acc = ((y_pred == labels).float().prod(dim=-1)).mean()
            self.log("val/task_subset_acc", subset_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        # Most important!!!
        with torch.no_grad():
            y_true_mc = labels.argmax(dim=-1)
            y_pred_mc = y_pred.argmax(dim=-1)
            multiclass_acc = (y_true_mc == y_pred_mc).float().mean()
            self.log("val/task_multiclass_acc", multiclass_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)




