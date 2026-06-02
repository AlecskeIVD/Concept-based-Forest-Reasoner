from itertools import product
import logging
from functools import reduce

import torch
from src.model.abstractModel import ConceptMemoryTrees
from .mnistEncoder import MNISTEncoder
from .mnist_dataset import addition_dataset
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from src.MNIST.mnistMLP import MNISTMLP
from src.MNIST.randomF import extractTensorValues

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

def extractAccuracyBestMainModel():
    depth = 5
    memorySize = 50
    lr = 0.001/2
    useDescendingLR = False
    EMB_SIZE = 128
    TREE_EMB_SIZE = EMB_SIZE
    TREEDECODERLAYERS = 1
    MAXEPOCHS = 70
    nConcepts = 20
    nOutputs = 19
    batchSize = 32
    maxDigit = 10
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

    accuracies = []
    for attempt in range(3):
        torch.manual_seed(42+attempt)
        predeterminedTrees = extractTensorValues(depth=depth, memorySize=memorySize, randomState=42+attempt, nConcepts=nConcepts, nOutputs=nOutputs)
        backBone = MNISTEncoder(emb_size=128, number_digits=2)
        # only 1 phase training
        model = ConceptMemoryTrees(backBone, EMB_SIZE, TREE_EMB_SIZE, TREEDECODERLAYERS, nConcepts, nOutputs, batchSize=batchSize, lr=lr, treeDepth=depth, memorySize=memorySize, dropout=False, dropoutP=0,preventFullyFalsePaths=False, distribution_weight=0, useDescendingLR=useDescendingLR, useALTLOSS=False, HashedSelector=False, treeLearner="predeterminedTrees", predeterminedTrees=True, givenTrees=predeterminedTrees)
        checkpoint_cb = ModelCheckpoint(dirpath="./model/MNIST/", save_top_k=1, monitor="val/loss", mode='min')        
        trainer = pl.Trainer(max_epochs=MAXEPOCHS, accelerator='auto', devices="auto", callbacks=[checkpoint_cb, EarlyStopping(monitor="val/loss", mode="min", patience=5)], enable_model_summary=False)
        # if disable validation step, no epoch completed
        trainer.fit(model=model, train_dataloaders=trainDl, val_dataloaders=valDl)
        model = ConceptMemoryTrees.load_from_checkpoint(checkpoint_cb.best_model_path, backbone=backBone, embeddingSize=EMB_SIZE, treeEmbeddingSize=TREE_EMB_SIZE, treeDecoderNbLayers=TREEDECODERLAYERS, nConcepts=nConcepts, nOutputs=nOutputs, batchSize=batchSize, lr=lr/10, treeDepth=depth, memorySize=memorySize, dropout=False, preventFullyFalsePaths=False, distribution_weight=0, hashedSelector=False, treeLearner="predeterminedTrees", predeterminedTrees=True, givenTrees=predeterminedTrees)
        results = trainer.test(model=model, dataloaders=testDl, verbose=False)
        test_accuracy = results[0]['test/task_multiclass_acc']
        accuracies.append(test_accuracy)
        print(f"Attempt {attempt+1}/3 for best config achieved test accuracy: {test_accuracy:.4f}")
    avgAccuracy = sum(accuracies)/len(accuracies)
    stdAccuracy = (sum([(x-avgAccuracy)**2 for x in accuracies])/len(accuracies))**0.5
    print(f"Final Test Accuracy: {avgAccuracy:.4f} ± {stdAccuracy:.4f} for best config: depth={depth}, memorySize={memorySize}, lr={lr}, useDescendingLR={useDescendingLR}")
    # Final Test Accuracy: 0.9447 ± 0.0079 for best config: depth=5, memorySize=50, lr=0.0005, useDescendingLR=False