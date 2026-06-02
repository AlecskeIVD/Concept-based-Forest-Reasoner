import numpy as np

from src.MNIST.mnist_dataset import addition_dataset
from itertools import product
import logging
from functools import reduce

import torch
from src.model.abstractModel import ConceptMemoryTrees
from .mnistEncoder import MNISTEncoder
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from src.MNIST.mnistMLP import MNISTMLP
from sklearn.metrics import r2_score


class MNISTMLPRegressor(pl.LightningModule):
    def __init__(self, encoder, embedding_size=256, num_classes=1):
        super().__init__()
        self.encoder = encoder
        self.fc1 = torch.nn.Linear(embedding_size, embedding_size)
        #self.fc2 = torch.nn.Linear(embedding_size, embedding_size)
        self.fc3 = torch.nn.Linear(embedding_size, num_classes)
        self.loss = torch.nn.MSELoss()
    

    def forward(self, x):
        x = self.encoder(x)
        x = torch.relu(self.fc1(x))
        x = self.fc3(x)
        x = x.squeeze(-1)
        return x
    

    def training_step(self, batch, batch_idx):
        x, c, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        self.log('train/loss', loss)
        # compute R2 score
        r2 = r2_score(y.detach().cpu().numpy(), logits.detach().cpu().numpy())
        self.log('train/r2', r2, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, c, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        self.log('val/loss', loss)
        # compute R2 score
        r2 = r2_score(y.detach().cpu().numpy(), logits.detach().cpu().numpy())
        self.log('val/r2', r2, prog_bar=True, on_step=False, on_epoch=True)
    

    def test_step(self, batch, batch_idx):
        x, c, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        self.log('test/loss', loss)
        # compute R2 score
        r2 = r2_score(y.detach().cpu().numpy(), logits.detach().cpu().numpy())
        self.log('test/r2', r2)
    

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


def createRegressionDataset(num_digits, digit_limit=10):
    X_train, c_train, y_train = addition_dataset(train=True, num_digits=num_digits, digit_limit=digit_limit)
    X_test, c_test, y_test = addition_dataset(train=False, num_digits=num_digits, digit_limit=digit_limit)
    X_train = torch.stack(X_train, dim=1)
    c_train = torch.cat(c_train, dim=-1)
    y_train = y_train.float()
    y_test = y_test.float()
    
    X_test = torch.stack(X_test, dim=1)
    c_test = torch.cat(c_test, dim=-1)

    print(X_train.shape, c_train.shape, y_train.shape)
    print(X_test.shape, c_test.shape, y_test.shape)

    return X_train, c_train, y_train, X_test, c_test, y_test


def trainRegressionModelMLP():
    MAXEPOCHS = 100
    torch.manual_seed(42)
    X_train, c_train, y_train, X_test, c_test, y_test = createRegressionDataset(num_digits=2)

    valSize = 0.1
    valSize = int(len(X_train) * valSize)
    trainSize = len(X_train) - valSize
    X_train, X_val = torch.split(X_train, [trainSize, valSize])
    c_train, c_val = torch.split(c_train, [trainSize, valSize])
    y_train, y_val = torch.split(y_train, [trainSize, valSize])

    trainDataset = TensorDataset(X_train, c_train, y_train)
    valDataset = TensorDataset(X_val, c_val, y_val)
    testDataset = TensorDataset(X_test, c_test, y_test)
    trainLoader = DataLoader(trainDataset, batch_size=64, shuffle=True, num_workers=4, persistent_workers=True)
    valLoader = DataLoader(valDataset, batch_size=64, shuffle=False, num_workers=4, persistent_workers=True)
    testLoader = DataLoader(testDataset, batch_size=len(X_test), shuffle=False, num_workers=4, persistent_workers=True)

    r2s = []
    mses = []
    for attempt in range(3):
        torch.manual_seed(42+attempt)
        backBone = MNISTEncoder(emb_size=128, number_digits=2)
        model = MNISTMLPRegressor(backBone, embedding_size=128, num_classes=1)
        checkpoint_cb = ModelCheckpoint(dirpath="./model/MNIST/", save_top_k=1, monitor="val/loss", mode='min')        
        trainer = pl.Trainer(max_epochs=MAXEPOCHS, accelerator='auto', devices="auto", callbacks=[checkpoint_cb], enable_model_summary=False)
        trainer.fit(model=model, train_dataloaders=trainLoader, val_dataloaders=valLoader)
        model = MNISTMLPRegressor.load_from_checkpoint(checkpoint_cb.best_model_path, encoder=backBone, embedding_size=128, num_classes=1)
        results = trainer.test(model=model, dataloaders=testLoader, verbose=False)
        val_accuracy = results[0]['test/r2']
        mse = results[0]['test/loss']
        r2s.append(val_accuracy)
        mses.append(mse)
    avgR2 = sum(r2s)/len(r2s)
    stdR2 = (sum([(x-avgR2)**2 for x in r2s])/len(r2s))**0.5
    avgMSE = sum(mses)/len(mses)
    stdMSE = (sum([(x-avgMSE)**2 for x in mses])/len(mses))**0.5
    print(f"MLP Regressor achieved R2-score: {avgR2:.4f} ± {stdR2:.4f} and MSE: {avgMSE:.4f} ± {stdMSE:.4f}")


def trainRegressionModelbase():
    torch.manual_seed(42)
    X_train, c_train, y_train, X_test, c_test, y_test = createRegressionDataset(num_digits=2)

    valSize = 0.1
    valSize = int(len(X_train) * valSize)
    trainSize = len(X_train) - valSize
    X_train, X_val = torch.split(X_train, [trainSize, valSize])
    c_train, c_val = torch.split(c_train, [trainSize, valSize])
    y_train, y_val = torch.split(y_train, [trainSize, valSize])

    trainDataset = TensorDataset(X_train, c_train, y_train)
    valDataset = TensorDataset(X_val, c_val, y_val)
    testDataset = TensorDataset(X_test, c_test, y_test)
    trainLoader = DataLoader(trainDataset, batch_size=64, shuffle=True, num_workers=4, persistent_workers=True)
    valLoader = DataLoader(valDataset, batch_size=64, shuffle=False, num_workers=4, persistent_workers=True)
    testLoader = DataLoader(testDataset, batch_size=len(X_test), shuffle=False, num_workers=4, persistent_workers=True)

    maxDigit = 10
    numberDigits = 2
    EMB_SIZE = 128
    TREE_EMB_SIZE = EMB_SIZE
    TREEDECODERLAYERS = 3
    nConcepts = numberDigits * maxDigit
    nOutputs = maxDigit * numberDigits - numberDigits + 1
    MAXEPOCHS = 100
    batchSize = 32
    depths = [5, 4, 3]
    memorySizes = [100, 50, 20]
    lrs = [ 0.001]
    descendingLR = [False]
    NUMOFTRIES = 3

    accuracies = []
    valAccuracies = []
    mses = []
    for depth, memorySize, lr, useDescendingLR in product(depths, memorySizes, lrs, descendingLR):
        resultsForConfig = []
        msesForConfig = []
        valAccuraciesForConfig = []
        for attempt in range(NUMOFTRIES):
            torch.manual_seed(42+attempt)
            backBone = MNISTEncoder(emb_size=128, number_digits=2)
            #PHASE 1: train with dropout max
            model = ConceptMemoryTrees(backBone, EMB_SIZE, TREE_EMB_SIZE, TREEDECODERLAYERS, nConcepts, nOutputs, batchSize=batchSize, lr=lr, treeDepth=depth, memorySize=memorySize, dropout=False, dropoutP=0,preventFullyFalsePaths=False, distribution_weight=0, useDescendingLR=useDescendingLR, useALTLOSS=False, HashedSelector=False, treeLearner="regressionFPT")
            checkpoint_cb = ModelCheckpoint(dirpath="./model/MNIST/", save_top_k=1, monitor="val/loss", mode='min')        
            trainer = pl.Trainer(max_epochs=MAXEPOCHS, accelerator='auto', devices="auto", callbacks=[checkpoint_cb, EarlyStopping(monitor="val/loss", mode="min", patience=5)], enable_model_summary=False)
            # if disable validation step, no epoch completed
            trainer.fit(model=model, train_dataloaders=trainLoader, val_dataloaders=valLoader)

            model = ConceptMemoryTrees.load_from_checkpoint(checkpoint_cb.best_model_path, backbone=backBone, embeddingSize=EMB_SIZE, treeEmbeddingSize=TREE_EMB_SIZE, treeDecoderNbLayers=TREEDECODERLAYERS, nConcepts=nConcepts, nOutputs=nOutputs, batchSize=batchSize, lr=lr/10, treeDepth=depth, memorySize=memorySize, dropout=False, preventFullyFalsePaths=False, distribution_weight=0, useDescendingLR=False, useALTLOSS=False, HashedSelector=False,  treeLearner="regressionFPT")
            results = trainer.test(model=model, dataloaders=valLoader, verbose=False)
            val_accuracy = results[0]['test/r2_score']
            valAccuraciesForConfig.append(val_accuracy)
            results = trainer.test(model=model, dataloaders=testLoader, verbose=False)
            val_accuracy = results[0]['test/r2_score']
            mse = results[0]['test/mse']
            resultsForConfig.append(val_accuracy)
            msesForConfig.append(mse)
            print(f"Attempt {attempt+1}/{NUMOFTRIES} for config depth={depth}, memorySize={memorySize}, lr={lr}, useDescendingLR={useDescendingLR} achieved validation accuracy: {val_accuracy:.4f} and mse: {mse:.4f}")
        
        avgAccuracy = sum(valAccuraciesForConfig)/len(valAccuraciesForConfig)
        stdAccuracy = (sum([(x-avgAccuracy)**2 for x in valAccuraciesForConfig])/len(valAccuraciesForConfig))**0.5
        valAccuracies.append((avgAccuracy, stdAccuracy, (depth, memorySize, lr, useDescendingLR)))
        avgAccuracy = sum(resultsForConfig)/len(resultsForConfig)
        stdAccuracy = (sum([(x-avgAccuracy)**2 for x in resultsForConfig])/len(resultsForConfig))**0.5
        accuracies.append((avgAccuracy, stdAccuracy, (depth, memorySize, lr, useDescendingLR)))
        avgMSE = sum(msesForConfig)/len(msesForConfig)
        stdMSE = (sum([(x-avgMSE)**2 for x in msesForConfig])/len(msesForConfig))**0.5
        mses.append((avgMSE, stdMSE, (depth, memorySize, lr, useDescendingLR)))
    valAccuracies.sort(key=lambda x: x[0], reverse=True)
    for acc in valAccuracies:
        print("VALIDATION")
        print(f"R2-score: {acc[0]:.4f} ± {acc[1]:.4f} for config: depth={acc[2][0]}, memorySize={acc[2][1]}, lr={acc[2][2]}, useDescendingLR={acc[2][3]}")
        for acc2 in accuracies:
            if acc2[2] == acc[2]:
                print(f"TEST R2-score: {acc2[0]:.4f} ± {acc2[1]:.4f}")
                break
        for mse in mses:
            if mse[2] == acc[2]:
                print(f"MSE: {mse[0]:.4f} ± {mse[1]:.4f}")
                break
        break


def trainRegressionModeldropout():
    torch.manual_seed(42)
    X_train, c_train, y_train, X_test, c_test, y_test = createRegressionDataset(num_digits=2)

    valSize = 0.1
    valSize = int(len(X_train) * valSize)
    trainSize = len(X_train) - valSize
    X_train, X_val = torch.split(X_train, [trainSize, valSize])
    c_train, c_val = torch.split(c_train, [trainSize, valSize])
    y_train, y_val = torch.split(y_train, [trainSize, valSize])

    trainDataset = TensorDataset(X_train, c_train, y_train)
    valDataset = TensorDataset(X_val, c_val, y_val)
    testDataset = TensorDataset(X_test, c_test, y_test)
    trainLoader = DataLoader(trainDataset, batch_size=64, shuffle=True, num_workers=4, persistent_workers=True)
    valLoader = DataLoader(valDataset, batch_size=64, shuffle=False, num_workers=4, persistent_workers=True)
    testLoader = DataLoader(testDataset, batch_size=len(X_test), shuffle=False, num_workers=4, persistent_workers=True)

    maxDigit = 10
    numberDigits = 2
    EMB_SIZE = 128
    TREE_EMB_SIZE = EMB_SIZE
    TREEDECODERLAYERS = 3
    nConcepts = numberDigits * maxDigit
    nOutputs = maxDigit * numberDigits - numberDigits + 1
    MAXEPOCHS = 100
    batchSize = 32
    depths = [5, 4, 3]
    memorySizes = [100, 50, 20]
    lrs = [ 0.001]
    descendingLR = [False]
    NUMOFTRIES = 3

    accuracies = []
    valAccuracies = []
    mses = []
    for depth, memorySize, lr, useDescendingLR in product(depths, memorySizes, lrs, descendingLR):
        resultsForConfig = []
        msesForConfig = []
        valAccuraciesForConfig = []
        for attempt in range(NUMOFTRIES):
            torch.manual_seed(42+attempt)
            backBone = MNISTEncoder(emb_size=128, number_digits=2)
            #PHASE 1: train with dropout max
            model = ConceptMemoryTrees(backBone, EMB_SIZE, TREE_EMB_SIZE, TREEDECODERLAYERS, nConcepts, nOutputs, batchSize=batchSize, lr=lr, treeDepth=depth, memorySize=memorySize, dropout=True, dropoutP=0.2,preventFullyFalsePaths=False, distribution_weight=0, useDescendingLR=useDescendingLR, useALTLOSS=False, HashedSelector=False, treeLearner="regressionFPT")
            checkpoint_cb = ModelCheckpoint(dirpath="./model/MNIST/", save_top_k=1, monitor="val/loss", mode='min')        
            trainer = pl.Trainer(max_epochs=MAXEPOCHS, accelerator='auto', devices="auto", callbacks=[checkpoint_cb, EarlyStopping(monitor="val/loss", mode="min", patience=5)], enable_model_summary=False)
            # if disable validation step, no epoch completed
            trainer.fit(model=model, train_dataloaders=trainLoader, val_dataloaders=valLoader)

            model = ConceptMemoryTrees.load_from_checkpoint(checkpoint_cb.best_model_path, backbone=backBone, embeddingSize=EMB_SIZE, treeEmbeddingSize=TREE_EMB_SIZE, treeDecoderNbLayers=TREEDECODERLAYERS, nConcepts=nConcepts, nOutputs=nOutputs, batchSize=batchSize, lr=lr/10, treeDepth=depth, memorySize=memorySize, dropout=False, preventFullyFalsePaths=False, distribution_weight=0, useDescendingLR=False, useALTLOSS=False, HashedSelector=False,  treeLearner="regressionFPT")
            results = trainer.test(model=model, dataloaders=valLoader, verbose=False)
            val_accuracy = results[0]['test/r2_score']
            valAccuraciesForConfig.append(val_accuracy)
            results = trainer.test(model=model, dataloaders=testLoader, verbose=False)
            val_accuracy = results[0]['test/r2_score']
            mse = results[0]['test/mse']
            resultsForConfig.append(val_accuracy)
            msesForConfig.append(mse)
            print(f"Attempt {attempt+1}/{NUMOFTRIES} for config depth={depth}, memorySize={memorySize}, lr={lr}, useDescendingLR={useDescendingLR} achieved validation accuracy: {val_accuracy:.4f} and mse: {mse:.4f}")
        
        avgAccuracy = sum(valAccuraciesForConfig)/len(valAccuraciesForConfig)
        stdAccuracy = (sum([(x-avgAccuracy)**2 for x in valAccuraciesForConfig])/len(valAccuraciesForConfig))**0.5
        valAccuracies.append((avgAccuracy, stdAccuracy, (depth, memorySize, lr, useDescendingLR)))
        avgAccuracy = sum(resultsForConfig)/len(resultsForConfig)
        stdAccuracy = (sum([(x-avgAccuracy)**2 for x in resultsForConfig])/len(resultsForConfig))**0.5
        accuracies.append((avgAccuracy, stdAccuracy, (depth, memorySize, lr, useDescendingLR)))
        avgMSE = sum(msesForConfig)/len(msesForConfig)
        stdMSE = (sum([(x-avgMSE)**2 for x in msesForConfig])/len(msesForConfig))**0.5
        mses.append((avgMSE, stdMSE, (depth, memorySize, lr, useDescendingLR)))
    valAccuracies.sort(key=lambda x: x[0], reverse=True)
    for acc in valAccuracies:
        print("VALIDATION")
        print(f"R2-score: {acc[0]:.4f} ± {acc[1]:.4f} for config: depth={acc[2][0]}, memorySize={acc[2][1]}, lr={acc[2][2]}, useDescendingLR={acc[2][3]}")
        for acc2 in accuracies:
            if acc2[2] == acc[2]:
                print(f"TEST R2-score: {acc2[0]:.4f} ± {acc2[1]:.4f}")
                break
        for mse in mses:
            if mse[2] == acc[2]:
                print(f"MSE: {mse[0]:.4f} ± {mse[1]:.4f}")
                break
        break

def trainRegressionModel2stage():
    torch.manual_seed(42)
    X_train, c_train, y_train, X_test, c_test, y_test = createRegressionDataset(num_digits=2)

    valSize = 0.1
    valSize = int(len(X_train) * valSize)
    trainSize = len(X_train) - valSize
    X_train, X_val = torch.split(X_train, [trainSize, valSize])
    c_train, c_val = torch.split(c_train, [trainSize, valSize])
    y_train, y_val = torch.split(y_train, [trainSize, valSize])

    trainDataset = TensorDataset(X_train, c_train, y_train)
    valDataset = TensorDataset(X_val, c_val, y_val)
    testDataset = TensorDataset(X_test, c_test, y_test)
    trainLoader = DataLoader(trainDataset, batch_size=64, shuffle=True, num_workers=4, persistent_workers=True)
    valLoader = DataLoader(valDataset, batch_size=64, shuffle=False, num_workers=4, persistent_workers=True)
    testLoader = DataLoader(testDataset, batch_size=len(X_test), shuffle=False, num_workers=4, persistent_workers=True)

    maxDigit = 10
    numberDigits = 2
    EMB_SIZE = 128
    TREE_EMB_SIZE = EMB_SIZE
    TREEDECODERLAYERS = 3
    nConcepts = numberDigits * maxDigit
    nOutputs = maxDigit * numberDigits - numberDigits + 1
    MAXEPOCHS = 100
    batchSize = 32
    depths = [5, 4, 3]
    memorySizes = [100, 50, 20]
    lrs = [ 0.001]
    descendingLR = [False]
    NUMOFTRIES = 3

    accuracies = []
    valAccuracies = []
    mses = []
    for depth, memorySize, lr, useDescendingLR in product(depths, memorySizes, lrs, descendingLR):
        resultsForConfig = []
        msesForConfig = []
        valAccuraciesForConfig = []
        for attempt in range(NUMOFTRIES):
            torch.manual_seed(42+attempt)
            backBone = MNISTEncoder(emb_size=128, number_digits=2)
            #PHASE 1: train with dropout max
            model = ConceptMemoryTrees(backBone, EMB_SIZE, TREE_EMB_SIZE, TREEDECODERLAYERS, nConcepts, nOutputs, batchSize=batchSize, lr=lr, treeDepth=depth, memorySize=memorySize, dropout=True, dropoutP=1,preventFullyFalsePaths=False, distribution_weight=0, useDescendingLR=useDescendingLR, useALTLOSS=False, HashedSelector=True, treeLearner="regressionFPT")
            checkpoint_cb = ModelCheckpoint(dirpath="./model/MNIST/", save_top_k=1, monitor="val/loss", mode='min')        
            trainer = pl.Trainer(max_epochs=MAXEPOCHS, accelerator='auto', devices="auto", callbacks=[checkpoint_cb, EarlyStopping(monitor="val/loss", mode="min", patience=5)], enable_model_summary=False)
            # if disable validation step, no epoch completed
            trainer.fit(model=model, train_dataloaders=trainLoader, val_dataloaders=valLoader)

            model = ConceptMemoryTrees.load_from_checkpoint(checkpoint_cb.best_model_path, backbone=backBone, embeddingSize=EMB_SIZE, treeEmbeddingSize=TREE_EMB_SIZE, treeDecoderNbLayers=TREEDECODERLAYERS, nConcepts=nConcepts, nOutputs=nOutputs, batchSize=batchSize, lr=lr/10, treeDepth=depth, memorySize=memorySize, dropout=False, preventFullyFalsePaths=False, distribution_weight=0, useDescendingLR=False, useALTLOSS=False, HashedSelector=False,  treeLearner="regressionFPT")
            model.NoSelector = False

            # PHASE 2: train without dropout, but initialize with the weights of the model trained with dropout
            checkpoint_cb = ModelCheckpoint(dirpath="./model/MNIST/", save_top_k=1, monitor="val/loss", mode='min')
            trainer = pl.Trainer(max_epochs=MAXEPOCHS, accelerator='auto', devices="auto", callbacks=[checkpoint_cb, EarlyStopping(monitor="val/loss", mode="min", patience=5)], enable_model_summary=False)
            trainer.fit(model=model, train_dataloaders=trainLoader, val_dataloaders=valLoader) 
            model = ConceptMemoryTrees.load_from_checkpoint(checkpoint_cb.best_model_path, backbone=backBone, embeddingSize=EMB_SIZE, treeEmbeddingSize=TREE_EMB_SIZE, treeDecoderNbLayers=TREEDECODERLAYERS, nConcepts=nConcepts, nOutputs=nOutputs, batchSize=batchSize, lr=lr/10, treeDepth=depth, memorySize=memorySize, dropout=False, preventFullyFalsePaths=False, distribution_weight=0, hashedSelector=False, treeLearner="regressionFPT")
            results = trainer.test(model=model, dataloaders=valLoader, verbose=False)
            val_accuracy = results[0]['test/r2_score']
            valAccuraciesForConfig.append(val_accuracy)
            results = trainer.test(model=model, dataloaders=testLoader, verbose=False)
            val_accuracy = results[0]['test/r2_score']
            mse = results[0]['test/mse']
            resultsForConfig.append(val_accuracy)
            msesForConfig.append(mse)
            print(f"Attempt {attempt+1}/{NUMOFTRIES} for config depth={depth}, memorySize={memorySize}, lr={lr}, useDescendingLR={useDescendingLR} achieved validation accuracy: {val_accuracy:.4f} and mse: {mse:.4f}")
        
        avgAccuracy = sum(valAccuraciesForConfig)/len(valAccuraciesForConfig)
        stdAccuracy = (sum([(x-avgAccuracy)**2 for x in valAccuraciesForConfig])/len(valAccuraciesForConfig))**0.5
        valAccuracies.append((avgAccuracy, stdAccuracy, (depth, memorySize, lr, useDescendingLR)))
        avgAccuracy = sum(resultsForConfig)/len(resultsForConfig)
        stdAccuracy = (sum([(x-avgAccuracy)**2 for x in resultsForConfig])/len(resultsForConfig))**0.5
        accuracies.append((avgAccuracy, stdAccuracy, (depth, memorySize, lr, useDescendingLR)))
        avgMSE = sum(msesForConfig)/len(msesForConfig)
        stdMSE = (sum([(x-avgMSE)**2 for x in msesForConfig])/len(msesForConfig))**0.5
        mses.append((avgMSE, stdMSE, (depth, memorySize, lr, useDescendingLR)))
    valAccuracies.sort(key=lambda x: x[0], reverse=True)
    for acc in valAccuracies:
        print("VALIDATION")
        print(f"R2-score: {acc[0]:.4f} ± {acc[1]:.4f} for config: depth={acc[2][0]}, memorySize={acc[2][1]}, lr={acc[2][2]}, useDescendingLR={acc[2][3]}")
        for acc2 in accuracies:
            if acc2[2] == acc[2]:
                print(f"TEST R2-score: {acc2[0]:.4f} ± {acc2[1]:.4f}")
                break
        for mse in mses:
            if mse[2] == acc[2]:
                print(f"MSE: {mse[0]:.4f} ± {mse[1]:.4f}")
                break
        break
    """R2-score: 0.9684 ± 0.0024 for config: depth=4, memorySize=100, lr=0.001, useDescendingLR=False
R2-score: 0.9672 ± 0.0021 for config: depth=3, memorySize=100, lr=0.001, useDescendingLR=False
R2-score: 0.9668 ± 0.0008 for config: depth=5, memorySize=20, lr=0.001, useDescendingLR=False
R2-score: 0.9660 ± 0.0018 for config: depth=5, memorySize=100, lr=0.001, useDescendingLR=False
R2-score: 0.9639 ± 0.0066 for config: depth=3, memorySize=20, lr=0.001, useDescendingLR=False
R2-score: 0.8301 ± 0.1942 for config: depth=4, memorySize=50, lr=0.001, useDescendingLR=False
R2-score: 0.7935 ± 0.2363 for config: depth=3, memorySize=50, lr=0.001, useDescendingLR=False
R2-score: 0.7734 ± 0.2721 for config: depth=5, memorySize=50, lr=0.001, useDescendingLR=False
R2-score: 0.3082 ± 0.9270 for config: depth=4, memorySize=20, lr=0.001, useDescendingLR=False"""


def scatterPlot():
    torch.manual_seed(42)
    X_train, c_train, y_train, X_test, c_test, y_test = createRegressionDataset(num_digits=2)

    valSize = 0.1
    valSize = int(len(X_train) * valSize)
    trainSize = len(X_train) - valSize
    X_train, X_val = torch.split(X_train, [trainSize, valSize])
    c_train, c_val = torch.split(c_train, [trainSize, valSize])
    y_train, y_val = torch.split(y_train, [trainSize, valSize])

    trainDataset = TensorDataset(X_train, c_train, y_train)
    valDataset = TensorDataset(X_val, c_val, y_val)
    testDataset = TensorDataset(X_test, c_test, y_test)
    trainLoader = DataLoader(trainDataset, batch_size=64, shuffle=True, drop_last=True, num_workers=4, persistent_workers=True)
    valLoader = DataLoader(valDataset, batch_size=64, shuffle=False, num_workers=4, persistent_workers=True)
    testLoader = DataLoader(testDataset, batch_size=len(X_test), shuffle=False)

    maxDigit = 10
    numberDigits = 2
    EMB_SIZE = 128
    TREE_EMB_SIZE = EMB_SIZE
    TREEDECODERLAYERS = 3
    nConcepts = numberDigits * maxDigit
    nOutputs = 1
    MAXEPOCHS = 100
    batchSize = 32
    depths = [4]
    memorySizes = [100]
    lrs = [ 0.001]
    descendingLR = [False]
    NUMOFTRIES = 1
    for depth, memorySize, lr, useDescendingLR in product(depths, memorySizes, lrs, descendingLR):
        for attempt in range(NUMOFTRIES):
            torch.manual_seed(42+attempt)
            backBone = MNISTEncoder(emb_size=128, number_digits=2)
            #PHASE 1: train with dropout max
            model = ConceptMemoryTrees(backBone, EMB_SIZE, TREE_EMB_SIZE, TREEDECODERLAYERS, nConcepts, nOutputs, batchSize=batchSize, lr=lr, treeDepth=depth, memorySize=memorySize, dropout=False, dropoutP=0.0,preventFullyFalsePaths=False, distribution_weight=0, useDescendingLR=useDescendingLR, useALTLOSS=False, HashedSelector=False, treeLearner="regressionFPT")
            checkpoint_cb = ModelCheckpoint(dirpath="./model/MNIST/", save_top_k=1, monitor="val/loss", mode='min')        
            trainer = pl.Trainer(max_epochs=MAXEPOCHS, accelerator='auto', devices="auto", callbacks=[checkpoint_cb, EarlyStopping(monitor="val/loss", mode="min", patience=5)], enable_model_summary=False)
            # if disable validation step, no epoch completed
            trainer.fit(model=model, train_dataloaders=trainLoader, val_dataloaders=valLoader)

            model = ConceptMemoryTrees.load_from_checkpoint(checkpoint_cb.best_model_path, backbone=backBone, embeddingSize=EMB_SIZE, treeEmbeddingSize=TREE_EMB_SIZE, treeDecoderNbLayers=TREEDECODERLAYERS, nConcepts=nConcepts, nOutputs=nOutputs, batchSize=batchSize, lr=lr/10, treeDepth=depth, memorySize=memorySize, dropout=False, preventFullyFalsePaths=False, distribution_weight=0, useDescendingLR=False, useALTLOSS=False, HashedSelector=False,  treeLearner="regressionFPT")
            results = trainer.test(model=model, dataloaders=testLoader, verbose=False)
            val_accuracy = results[0]['test/r2_score']
            print(f"Attempt {attempt+1}/{NUMOFTRIES} for config depth={depth}, memorySize={memorySize}, lr={lr}, useDescendingLR={useDescendingLR} achieved validation accuracy: {val_accuracy:.4f}")
            
            # iterate over the test set and collect the predictions and true values
            model.eval()
            all_predictions = []
            all_true_values = []
            all_concepts = []
            all_predicted_concepts = []
            with torch.no_grad():
                for batch in testLoader:
                    X_batch, c_batch, y_batch = batch
                    fw = model(batch)
                    predictions = fw["y_pred_probs_bo"]
                    c_pred_probs_bc = fw["c_pred_probs_bc"]
                    c_pred = (c_pred_probs_bc > 0.5).float()
                    all_predictions.extend(predictions.cpu().numpy())
                    all_true_values.extend(y_batch.cpu().numpy())
                    all_concepts.extend(c_batch.cpu().float().numpy())
                    all_predicted_concepts.extend(c_pred.cpu().numpy())
                
                # make a scatter plot of the predictions vs the true values
                # add random noise to the predictions and labels for better visualization
                all_predictions = np.array(all_predictions) + np.random.normal(0, 0.2, len(all_predictions))
                all_true_values = np.array(all_true_values) + np.random.normal(0, 0.2, len(all_true_values))

                # if predicted concepts are correct, color the point green, otherwise red
                all_concepts = np.array(all_concepts)
                all_predicted_concepts = np.array(all_predicted_concepts)
                correct_concepts = np.all(all_concepts == all_predicted_concepts, axis=1)
                colors = ['green' if correct else 'red' for correct in correct_concepts]

                import matplotlib.pyplot as plt
                plt.scatter(all_true_values, all_predictions, c=colors, alpha=0.5)
                plt.xlabel("True Values")
                plt.ylabel("Predictions")
                plt.title(f"Predictions vs True Values for MLP")
                plt.plot([min(all_true_values), max(all_true_values)], [min(all_true_values), max(all_true_values)], 'r--')  # line y=x for reference
                plt.show()
        


if __name__ == "__main__":
    #createRegressionDataset(num_digits=2)
    scatterPlot()