from itertools import product
import logging
from functools import reduce

from matplotlib import pyplot as plt
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

def baseTrees():
    #HYPER
    maxDigit = 10
    numberDigits = 2
    EMB_SIZE = 128
    TREE_EMB_SIZE = EMB_SIZE
    TREEDECODERLAYERS = 3
    nConcepts = numberDigits * maxDigit
    nOutputs = maxDigit * numberDigits - numberDigits + 1
    MAXEPOCHS = 100
    #MAXEPOCHS = 1
    batchSize = 32
    depths = [3]
    #depths = [5]
    memorySizes = [50]
    #memorySizes = [100]
    lrs = [ 0.001]
    descendingLR = [False]
    NUMOFTRIES = 1
    
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
    testAccuracies = []
    for depth, memorySize, lr, useDescendingLR in product(depths, memorySizes, lrs, descendingLR):
        resultsForConfig = []
        testResultsForConfig = []
        for attempt in range(NUMOFTRIES):
            torch.manual_seed(42+attempt)
            backBone = MNISTEncoder(emb_size=128, number_digits=2)
            #PHASE 1: train with dropout max
            model = ConceptMemoryTrees(backBone, EMB_SIZE, TREE_EMB_SIZE, TREEDECODERLAYERS, nConcepts, nOutputs, batchSize=batchSize, lr=lr, treeDepth=depth, memorySize=memorySize, dropout=False, dropoutP=0,preventFullyFalsePaths=False, distribution_weight=0, useDescendingLR=useDescendingLR, useALTLOSS=False, HashedSelector=False, treeLearner="indeptendentFPT")
            checkpoint_cb = ModelCheckpoint(dirpath="./model/MNIST/", save_top_k=1, monitor="val/loss", mode='min')        
            trainer = pl.Trainer(max_epochs=MAXEPOCHS, accelerator='auto', devices="auto", callbacks=[checkpoint_cb, EarlyStopping(monitor="val/loss", mode="min", patience=5)], enable_model_summary=False)
            # if disable validation step, no epoch completed
            trainer.fit(model=model, train_dataloaders=trainDl, val_dataloaders=valDl)
            model = ConceptMemoryTrees.load_from_checkpoint(checkpoint_cb.best_model_path, backbone=backBone, embeddingSize=EMB_SIZE, treeEmbeddingSize=TREE_EMB_SIZE, treeDecoderNbLayers=TREEDECODERLAYERS, nConcepts=nConcepts, nOutputs=nOutputs, batchSize=batchSize, lr=lr/10, treeDepth=depth, memorySize=memorySize, dropout=False, preventFullyFalsePaths=False, distribution_weight=0, hashedSelector=False, treeLearner="indeptendentFPT")

    #out = model.printTrees()
    #with open("/Users/alecvandeuren/Thesis/resultsFetchTrees.txt", "a") as f:
    #    f.write("Result for basic\n")
    #    f.write(out)
    testDl = DataLoader(TensorDataset(x_test, c_test, y_test), batch_size=1,  num_workers=7, persistent_workers=True)
    hists = model.collectHistoLeafs(testDl)
    plt.bar(range(len(hists)), hists)
    plt.xlabel("Leaf Index")
    plt.ylabel("Frequency")
    plt.title("Distribution of Most Likely Leaves for Standard CFR")
    plt.savefig("distribution_of_most_likely_leaves_standard.png")

def dropoutTrees():
    #HYPER
    maxDigit = 10
    numberDigits = 2
    EMB_SIZE = 128
    TREE_EMB_SIZE = EMB_SIZE
    TREEDECODERLAYERS = 3
    nConcepts = numberDigits * maxDigit
    nOutputs = maxDigit * numberDigits - numberDigits + 1
    MAXEPOCHS = 100
    #MAXEPOCHS = 1
    batchSize = 32
    depths = [3]
    #depths = [5]
    memorySizes = [50]
    #memorySizes = [100]
    lrs = [ 0.001]
    descendingLR = [False]
    NUMOFTRIES = 1
    
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
    testAccuracies = []
    for depth, memorySize, lr, useDescendingLR in product(depths, memorySizes, lrs, descendingLR):
        resultsForConfig = []
        testResultsForConfig = []
        for attempt in range(NUMOFTRIES):
            torch.manual_seed(42+attempt)
            backBone = MNISTEncoder(emb_size=128, number_digits=2)
            #PHASE 1: train with dropout max
            model = ConceptMemoryTrees(backBone, EMB_SIZE, TREE_EMB_SIZE, TREEDECODERLAYERS, nConcepts, nOutputs, batchSize=batchSize, lr=lr, treeDepth=depth, memorySize=memorySize, dropout=True, dropoutP=0.2,preventFullyFalsePaths=False, distribution_weight=0, useDescendingLR=useDescendingLR, useALTLOSS=False, HashedSelector=False, treeLearner="indeptendentFPT")
            checkpoint_cb = ModelCheckpoint(dirpath="./model/MNIST/", save_top_k=1, monitor="val/loss", mode='min')        
            trainer = pl.Trainer(max_epochs=MAXEPOCHS, accelerator='auto', devices="auto", callbacks=[checkpoint_cb, EarlyStopping(monitor="val/loss", mode="min", patience=5)], enable_model_summary=False)
            # if disable validation step, no epoch completed
            trainer.fit(model=model, train_dataloaders=trainDl, val_dataloaders=valDl)
            model = ConceptMemoryTrees.load_from_checkpoint(checkpoint_cb.best_model_path, backbone=backBone, embeddingSize=EMB_SIZE, treeEmbeddingSize=TREE_EMB_SIZE, treeDecoderNbLayers=TREEDECODERLAYERS, nConcepts=nConcepts, nOutputs=nOutputs, batchSize=batchSize, lr=lr/10, treeDepth=depth, memorySize=memorySize, dropout=False, preventFullyFalsePaths=False, distribution_weight=0, hashedSelector=False, treeLearner="indeptendentFPT")

    #out = model.printTrees()
    #with open("/Users/alecvandeuren/Thesis/resultsFetchTrees.txt", "a") as f:
    #    f.write("Result for DROPOUT\n")
    #    f.write(out)
    testDl = DataLoader(TensorDataset(x_test, c_test, y_test), batch_size=1,  num_workers=7, persistent_workers=True)
    hists = model.collectHistoLeafs(testDl)
    plt.bar(range(len(hists)), hists)
    plt.xlabel("Leaf Index")
    plt.ylabel("Frequency")
    plt.title("Distribution of Most Likely Leaves for Dropout CFR")
    plt.savefig("distribution_of_most_likely_leaves_dropout.png")
    plt.clf()


def twoStage():
    results = """Accuracy: 0.9624 ± 0.0072 for config: depth=5, memorySize=50, lr=0.001, useDescendingLR=False
Accuracy: 0.9579 ± 0.0097 for config: depth=5, memorySize=100, lr=0.001, useDescendingLR=False
Accuracy: 0.9568 ± 0.0076 for config: depth=4, memorySize=100, lr=0.001, useDescendingLR=False
Accuracy: 0.9508 ± 0.0082 for config: depth=3, memorySize=100, lr=0.001, useDescendingLR=False
Accuracy: 0.9502 ± 0.0104 for config: depth=4, memorySize=50, lr=0.001, useDescendingLR=False
Accuracy: 0.9274 ± 0.0352 for config: depth=5, memorySize=20, lr=0.001, useDescendingLR=False
Accuracy: 0.9087 ± 0.0188 for config: depth=3, memorySize=50, lr=0.001, useDescendingLR=False
Accuracy: 0.8899 ± 0.0449 for config: depth=4, memorySize=20, lr=0.001, useDescendingLR=False
Accuracy: 0.8486 ± 0.0218 for config: depth=3, memorySize=20, lr=0.001, useDescendingLR=False"""
    depth = 3
    memorySize = 50
    lr = 0.001
    useDescendingLR = False
    EMB_SIZE = 128
    TREE_EMB_SIZE = EMB_SIZE
    TREEDECODERLAYERS = 1
    MAXEPOCHS = 50
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
    for attempt in range(1):
        torch.manual_seed(42+attempt)
        backBone = MNISTEncoder(emb_size=128, number_digits=2)
        #PHASE 1: train with dropout max
        model = ConceptMemoryTrees(backBone, EMB_SIZE, TREE_EMB_SIZE, TREEDECODERLAYERS, nConcepts, nOutputs, batchSize=batchSize, lr=lr, treeDepth=depth, memorySize=memorySize, dropout=True, dropoutP=1,preventFullyFalsePaths=False, distribution_weight=0, useDescendingLR=useDescendingLR, useALTLOSS=False, HashedSelector=True, treeLearner="indeptendentFPT")
        checkpoint_cb = ModelCheckpoint(dirpath="./model/MNIST/", save_top_k=1, monitor="val/loss", mode='min')        
        trainer = pl.Trainer(max_epochs=MAXEPOCHS, accelerator='auto', devices="auto", callbacks=[checkpoint_cb, EarlyStopping(monitor="val/loss", mode="min", patience=5)], enable_model_summary=False)
        # if disable validation step, no epoch completed
        trainer.fit(model=model, train_dataloaders=trainDl, val_dataloaders=valDl)
        model = ConceptMemoryTrees.load_from_checkpoint(checkpoint_cb.best_model_path, backbone=backBone, embeddingSize=EMB_SIZE, treeEmbeddingSize=TREE_EMB_SIZE, treeDecoderNbLayers=TREEDECODERLAYERS, nConcepts=nConcepts, nOutputs=nOutputs, batchSize=batchSize, lr=lr/10, treeDepth=depth, memorySize=memorySize, dropout=False, preventFullyFalsePaths=False, distribution_weight=0, useDescendingLR=False, useALTLOSS=False, HashedSelector=False,  treeLearner="indeptendentFPT")
        model.NoSelector = False
        # PHASE 2: train without dropout, but initialize with the weights of the model trained with dropout
        checkpoint_cb = ModelCheckpoint(dirpath="./model/MNIST/", save_top_k=1, monitor="val/loss", mode='min')
        trainer = pl.Trainer(max_epochs=MAXEPOCHS, accelerator='auto', devices="auto", callbacks=[checkpoint_cb, EarlyStopping(monitor="val/loss", mode="min", patience=5)], enable_model_summary=False)
        trainer.fit(model=model, train_dataloaders=trainDl, val_dataloaders=valDl) 
        model = ConceptMemoryTrees.load_from_checkpoint(checkpoint_cb.best_model_path, backbone=backBone, embeddingSize=EMB_SIZE, treeEmbeddingSize=TREE_EMB_SIZE, treeDecoderNbLayers=TREEDECODERLAYERS, nConcepts=nConcepts, nOutputs=nOutputs, batchSize=batchSize, lr=lr/10, treeDepth=depth, memorySize=memorySize, dropout=False, preventFullyFalsePaths=False, distribution_weight=0, hashedSelector=False, treeLearner="indeptendentFPT")
    #trees = model.printTrees()
    #with open("/Users/alecvandeuren/Thesis/resultsFetchTrees.txt", "a") as f:
    #    f.write("Result for TWO STAGE TRAINING:\n")
    #    f.write(trees)

    testDl = DataLoader(TensorDataset(x_test, c_test, y_test), batch_size=1,  num_workers=7, persistent_workers=True)
    hists = model.collectHistoLeafs(testDl)
    plt.bar(range(len(hists)), hists)
    plt.xlabel("Leaf Index")
    plt.ylabel("Frequency")
    plt.title("Distribution of Most Likely Leaves for Two Stage Training")
    plt.savefig("distribution_of_most_likely_leaves_two_stage.png")
    plt.clf()