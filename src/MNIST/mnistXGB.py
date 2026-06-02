import torch
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from .mnistEncoder import MNISTEncoder
from .mnist_dataset import addition_dataset
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.functional import binary_cross_entropy_with_logits
from src.CUB.cubEncoder import CUBEncoder, LinearEncoder
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import logging
from itertools import product

from src.CUB.cubDataset import SELECTED_CONCEPTS

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


class mnistConceptPredictor(pl.LightningModule):
    def __init__(self, encoder, embedding_size=2048, num_classes=50):
        super(mnistConceptPredictor, self).__init__()
        self.encoder = encoder
        self.embeddingSize = embedding_size
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.embeddingSize, self.embeddingSize),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embeddingSize, self.embeddingSize),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embeddingSize, self.embeddingSize),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embeddingSize, num_classes),# size batchSize, nConcepts
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, c, y = batch
        logits = self.forward(x)
        loss = binary_cross_entropy_with_logits(logits, c.float())
        self.log('train/loss', loss)
        acc = ((logits > 0) == c).float().mean()
        self.log('train/acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, c, y = batch
        logits = self.forward(x)
        loss = binary_cross_entropy_with_logits(logits, c.float())
        self.log('val/loss', loss)
        acc = ((logits > 0) == c).float().mean()
        self.log('val/acc', acc, prog_bar=True, on_step=False, on_epoch=True)
    

    def test_step(self, batch, batch_idx):
        x, c, y = batch
        logits = self.forward(x)
        acc = ((logits > 0) == c).float().mean()
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
            }
        }
    
    def extractConceptPreds(self, dataloader):
        self.eval()
        all_preds = []

        with torch.no_grad():
            for x, c, y in dataloader:
                logits = self.forward(x.to(self.device))
                preds = (logits > 0).cpu().numpy()
                all_preds.append(preds)
        return np.vstack(all_preds)

def extract_numpy_data(dataloader):
    """
    Iterates through a PyTorch DataLoader and extracts the features and labels 
    into standard NumPy arrays for scikit-learn / XGBoost.
    """
    X_list = []
    y_list = []
    
    for embeddings, concepts, labels in dataloader:
        # We will use the embeddings as our features. 
        # (Alternatively, you could use: torch.cat((embeddings, concepts), dim=1) if you want both!)
        X_list.append(concepts.cpu().numpy())

        class_indices = torch.argmax(labels, dim=-1)
        y_list.append(class_indices.cpu().numpy())
        
    # Stack all batches into one giant 2D array
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    print(f"Extracted {X.shape[0]} samples with {X.shape[1]} features each.")
    print(f"Extracted {len(y)} labels with {len(np.unique(y))} unique classes.")
    return X, y

def runXGBExperiment():
    EMB_SIZE = 128
    nConcepts = 20
    nOutputs = 19
    batchSize = 32
    depths = [5, 4, 3]
    memorySizes = [100, 50, 20]
    maxEpochs = 50
    maxDigit = 10
    torch.manual_seed(42) 
    x_train, c_train, y_train = addition_dataset(True, 2, maxDigit)
    x_test, c_test, y_test = addition_dataset(False, 2, maxDigit)
    x_train = torch.stack(x_train, dim=1)
    c_train = torch.cat(c_train, dim=-1)
    y_train = F.one_hot(y_train.unsqueeze(-1).long().ravel()).float()

    val_split = 0.1
    train_set_size = int(len(x_train) * (1 - val_split))
    x_val, c_val, y_val = x_train[train_set_size:], c_train[train_set_size:], y_train[train_set_size:]
    x_train, c_train, y_train = x_train[:train_set_size], c_train[:train_set_size], y_train[:train_set_size]

    x_test = torch.stack(x_test, dim=1)
    c_test = torch.cat(c_test, dim=-1)
    y_test = F.one_hot(y_test.unsqueeze(-1).long().ravel()).float()
    trainDl = DataLoader(TensorDataset(x_train, c_train, y_train), batch_size=batchSize, shuffle=False, num_workers=7, persistent_workers=True)
    testDl = DataLoader(TensorDataset(x_test, c_test, y_test), batch_size=batchSize,  num_workers=7, persistent_workers=True)
    valDl = DataLoader(TensorDataset(x_val, c_val, y_val), batch_size=batchSize,  num_workers=7, persistent_workers=True)

    X_train, y_train = extract_numpy_data(trainDl)

    X_val, y_val = extract_numpy_data(valDl)

    #X_test, y_test = extract_numpy_data(test_loader)
    allAccuracies = []
    for depth, mem in product(depths, memorySizes):
        accuracies = []
        for trial in range(3):
            print(f"\nRunning XGBoost with max_depth={depth}, max_leaves={mem}, trial {trial+1}/3")
            torch.manual_seed(42 + trial)
            xgb_model = xgb.XGBClassifier(
                n_estimators=mem,      
                max_depth=depth,           
                learning_rate=0.1,     
                objective='multi:softmax', 
                tree_method='hist',    
                n_jobs=-1,
                early_stopping_rounds=15,
                random_state=42 + trial
            )

            xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)], 
                verbose=10                 
            )
            encoder = LinearEncoder(512, EMB_SIZE)
            encoder = MNISTEncoder(EMB_SIZE, number_digits=2)
            conceptPredictor = mnistConceptPredictor(encoder=encoder, embedding_size=EMB_SIZE, num_classes=nConcepts)
            checkpoint_cb = ModelCheckpoint(dirpath="./model/CUB/", save_top_k=1, monitor="val/loss", mode='min')        
            trainer = pl.Trainer(max_epochs=maxEpochs, accelerator='auto', devices="auto", callbacks=[checkpoint_cb, EarlyStopping(monitor="val/loss", mode="min", patience=5)], enable_model_summary=False)
            trainer.fit(model=conceptPredictor, train_dataloaders=trainDl, val_dataloaders=valDl)
            conceptPredictor = mnistConceptPredictor.load_from_checkpoint(checkpoint_cb.best_model_path, encoder=encoder, embedding_size=EMB_SIZE, num_classes=nConcepts)
            cPreds = conceptPredictor.extractConceptPreds(valDl)
            print(f"Extracted concept predictions shape: {cPreds.shape}")
            yPreds = xgb_model.predict(cPreds)
            acc = accuracy_score(y_val, yPreds)
            print(f"Validation Accuracy for XGBoost with max_depth={depth}, max_leaves={mem}, trial {trial+1}/3: {acc * 100:.2f}%")
            accuracies.append(acc)
        avg_acc = np.mean(accuracies)
        stdAccuracy = (sum([(x-avg_acc)**2 for x in accuracies])/len(accuracies))**0.5
        allAccuracies.append((avg_acc, stdAccuracy, (depth, mem)))
        print(f"\nAverage Validation Accuracy for max_depth={depth}, max_leaves={mem}: {avg_acc * 100:.2f}% ± {stdAccuracy * 100:.2f}%")
    print("\nSummary of all configurations:")
    allAccuracies.sort(key=lambda x: x[0]-x[1], reverse=True)
    for avg_acc, std_acc, (depth, mem) in allAccuracies:
        print(f"max_depth={depth}, max_trees={mem}: {avg_acc * 100:.2f}% ± {std_acc * 100:.2f}%")
"""
max_depth=4, max_trees=100: 93.12% ± 1.19%
max_depth=5, max_trees=100: 93.01% ± 1.22%
max_depth=5, max_trees=50: 75.23% ± 1.03%
max_depth=4, max_trees=50: 72.00% ± 0.87%
max_depth=3, max_trees=100: 70.66% ± 0.80%
max_depth=4, max_trees=20: 61.12% ± 0.92%
max_depth=5, max_trees=20: 59.74% ± 1.01%
max_depth=3, max_trees=50: 58.78% ± 0.82%
max_depth=3, max_trees=20: 45.82% ± 0.49%"""

def ComputeAccuracy():
    EMB_SIZE = 128
    nConcepts = 20
    nOutputs = 19
    batchSize = 32
    depths = [4]
    memorySizes = [100]
    maxEpochs = 50
    maxDigit = 10
    torch.manual_seed(42) 
    x_train, c_train, y_train = addition_dataset(True, 2, maxDigit)
    x_test, c_test, y_test = addition_dataset(False, 2, maxDigit)
    x_train = torch.stack(x_train, dim=1)
    c_train = torch.cat(c_train, dim=-1)
    y_train = F.one_hot(y_train.unsqueeze(-1).long().ravel()).float()

    val_split = 0.1
    train_set_size = int(len(x_train) * (1 - val_split))
    x_val, c_val, y_val = x_train[train_set_size:], c_train[train_set_size:], y_train[train_set_size:]
    x_train, c_train, y_train = x_train[:train_set_size], c_train[:train_set_size], y_train[:train_set_size]

    x_test = torch.stack(x_test, dim=1)
    c_test = torch.cat(c_test, dim=-1)
    y_test = F.one_hot(y_test.unsqueeze(-1).long().ravel()).float()
    trainDl = DataLoader(TensorDataset(x_train, c_train, y_train), batch_size=batchSize, shuffle=False, num_workers=7, persistent_workers=True)
    testDl = DataLoader(TensorDataset(x_test, c_test, y_test), batch_size=batchSize,  num_workers=7, persistent_workers=True)
    valDl = DataLoader(TensorDataset(x_val, c_val, y_val), batch_size=batchSize,  num_workers=7, persistent_workers=True)

    X_train, y_train = extract_numpy_data(trainDl)

    X_val, y_val = extract_numpy_data(valDl)

    X_test, y_test = extract_numpy_data(testDl)
    allAccuracies = []
    for depth, mem in product(depths, memorySizes):
        accuracies = []
        for trial in range(3):
            print(f"\nRunning XGBoost with max_depth={depth}, max_leaves={mem}, trial {trial+1}/3")
            torch.manual_seed(42 + trial)
            xgb_model = xgb.XGBClassifier(
                n_estimators=mem,      
                max_depth=depth,           
                learning_rate=0.1,     
                objective='multi:softmax', 
                tree_method='hist',    
                n_jobs=-1,
                early_stopping_rounds=15,
                random_state=42 + trial
            )

            xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)], 
                verbose=10                 
            )
            encoder = LinearEncoder(512, EMB_SIZE)
            encoder = MNISTEncoder(EMB_SIZE, number_digits=2)
            conceptPredictor = mnistConceptPredictor(encoder=encoder, embedding_size=EMB_SIZE, num_classes=nConcepts)
            checkpoint_cb = ModelCheckpoint(dirpath="./model/CUB/", save_top_k=1, monitor="val/loss", mode='min')        
            trainer = pl.Trainer(max_epochs=maxEpochs, accelerator='auto', devices="auto", callbacks=[checkpoint_cb, EarlyStopping(monitor="val/loss", mode="min", patience=5)], enable_model_summary=False)
            trainer.fit(model=conceptPredictor, train_dataloaders=trainDl, val_dataloaders=valDl)
            conceptPredictor = mnistConceptPredictor.load_from_checkpoint(checkpoint_cb.best_model_path, encoder=encoder, embedding_size=EMB_SIZE, num_classes=nConcepts)
            cPreds = conceptPredictor.extractConceptPreds(testDl)
            print(f"Extracted concept predictions shape: {cPreds.shape}")
            yPreds = xgb_model.predict(cPreds)
            acc = accuracy_score(y_test, yPreds)
            print(f"test Accuracy for XGBoost with max_depth={depth}, max_leaves={mem}, trial {trial+1}/3: {acc * 100:.2f}%")
            accuracies.append(acc)
        avg_acc = np.mean(accuracies)
        stdAccuracy = (sum([(x-avg_acc)**2 for x in accuracies])/len(accuracies))**0.5
        allAccuracies.append((avg_acc, stdAccuracy, (depth, mem)))
        print(f"\nAverage test Accuracy for max_depth={depth}, max_leaves={mem}: {avg_acc * 100:.2f}% ± {stdAccuracy * 100:.2f}%")
    print("\nSummary of all configurations:")
    allAccuracies.sort(key=lambda x: x[0]-x[1], reverse=True)
    for avg_acc, std_acc, (depth, mem) in allAccuracies:
        print(f"max_depth={depth}, max_trees={mem}: {avg_acc * 100:.2f}% ± {std_acc * 100:.2f}%")
    # Average test Accuracy for max_depth=4, max_leaves=100: 93.07% ± 1.05%
    
            

