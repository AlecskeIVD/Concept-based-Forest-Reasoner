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

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


def computeAccuracyMainModel():
    #HYPER
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
    for depth, memorySize, lr, useDescendingLR in product(depths, memorySizes, lrs, descendingLR):
        resultsForConfig = []
        for attempt in range(NUMOFTRIES):
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
            results = trainer.test(model=model, dataloaders=valDl, verbose=False)
            val_accuracy = results[0]['test/task_multiclass_acc']
            resultsForConfig.append(val_accuracy)
            print(f"Attempt {attempt+1}/{NUMOFTRIES} for config depth={depth}, memorySize={memorySize}, lr={lr}, useDescendingLR={useDescendingLR} achieved validation accuracy: {val_accuracy:.4f}")
        
        avgAccuracy = sum(resultsForConfig)/len(resultsForConfig)
        stdAccuracy = (sum([(x-avgAccuracy)**2 for x in resultsForConfig])/len(resultsForConfig))**0.5
        accuracies.append((avgAccuracy, stdAccuracy, (depth, memorySize, lr, useDescendingLR)))
    accuracies.sort(key=lambda x: x[0], reverse=True)
    for acc in accuracies:
        print(f"Accuracy: {acc[0]:.4f} ± {acc[1]:.4f} for config: depth={acc[2][0]}, memorySize={acc[2][1]}, lr={acc[2][2]}, useDescendingLR={acc[2][3]}")


def extractAccuracyBestMainModel():
    results = """Accuracy: 0.9624 ± 0.0072 for config: depth=5, memorySize=50, lr=0.001, useDescendingLR=False
Accuracy: 0.9579 ± 0.0097 for config: depth=5, memorySize=100, lr=0.001, useDescendingLR=False
Accuracy: 0.9568 ± 0.0076 for config: depth=4, memorySize=100, lr=0.001, useDescendingLR=False
Accuracy: 0.9508 ± 0.0082 for config: depth=3, memorySize=100, lr=0.001, useDescendingLR=False
Accuracy: 0.9502 ± 0.0104 for config: depth=4, memorySize=50, lr=0.001, useDescendingLR=False
Accuracy: 0.9274 ± 0.0352 for config: depth=5, memorySize=20, lr=0.001, useDescendingLR=False
Accuracy: 0.9087 ± 0.0188 for config: depth=3, memorySize=50, lr=0.001, useDescendingLR=False
Accuracy: 0.8899 ± 0.0449 for config: depth=4, memorySize=20, lr=0.001, useDescendingLR=False
Accuracy: 0.8486 ± 0.0218 for config: depth=3, memorySize=20, lr=0.001, useDescendingLR=False"""
    bestPerformer = results.split("\n")[0]
    depth = 5
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
    for attempt in range(3):
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
        results = trainer.test(model=model, dataloaders=testDl, verbose=False)
        test_accuracy = results[0]['test/task_multiclass_acc']
        accuracies.append(test_accuracy)
        print(f"Attempt {attempt+1}/3 for best config achieved test accuracy: {test_accuracy:.4f}")
    avgAccuracy = sum(accuracies)/len(accuracies)
    stdAccuracy = (sum([(x-avgAccuracy)**2 for x in accuracies])/len(accuracies))**0.5
    print(f"Final Test Accuracy: {avgAccuracy:.4f} ± {stdAccuracy:.4f} for best config: depth={depth}, memorySize={memorySize}, lr={lr}, useDescendingLR={useDescendingLR}")
    # Final Test Accuracy: 0.9613 ± 0.0053 for best config: depth=5, memorySize=50, lr=0.001, useDescendingLR=False

def MLPExperiment():
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
    for attempt in range(3):
        torch.manual_seed(42+attempt)
        encoder = MNISTEncoder(emb_size=128, number_digits=2)
        model = MNISTMLP(encoder, embedding_size=EMB_SIZE, num_classes=nOutputs)
        checkpoint_cb = ModelCheckpoint(dirpath="./model/MNIST_MLP/", save_top_k=1, monitor="val/loss", mode='min')        
        trainer = pl.Trainer(max_epochs=MAXEPOCHS, accelerator='auto', devices="auto", callbacks=[checkpoint_cb, EarlyStopping(monitor="val/loss", mode="min", patience=5)], enable_model_summary=False)
        trainer.fit(model=model, train_dataloaders=trainDl, val_dataloaders=valDl)
        model = MNISTMLP.load_from_checkpoint(checkpoint_cb.best_model_path, encoder=encoder, embedding_size=EMB_SIZE, num_classes=nOutputs)
        results = trainer.test(model=model, dataloaders=valDl, verbose=False)
        test_accuracy = results[0]['test/acc']
        accuracies.append(test_accuracy)
        print(f"Attempt {attempt+1}/3 for MLP achieved validation accuracy: {test_accuracy:.4f}")
    avgAccuracy = sum(accuracies)/len(accuracies)
    stdAccuracy = (sum([(x-avgAccuracy)**2 for x in accuracies])/len(accuracies))**0.5
    print(f"Final Validation Accuracy for MLP: {avgAccuracy:.4f} ± {stdAccuracy:.4f}")

def MLPComputeFinalAccuracy():
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
    for attempt in range(3):
        torch.manual_seed(42+attempt)
        encoder = MNISTEncoder(emb_size=128, number_digits=2)
        model = MNISTMLP(encoder, embedding_size=EMB_SIZE, num_classes=nOutputs)
        checkpoint_cb = ModelCheckpoint(dirpath="./model/MNIST_MLP/", save_top_k=1, monitor="val/loss", mode='min')        
        trainer = pl.Trainer(max_epochs=MAXEPOCHS, accelerator='auto', devices="auto", callbacks=[checkpoint_cb, EarlyStopping(monitor="val/loss", mode="min", patience=5)], enable_model_summary=False)
        trainer.fit(model=model, train_dataloaders=trainDl, val_dataloaders=valDl)
        model = MNISTMLP.load_from_checkpoint(checkpoint_cb.best_model_path, encoder=encoder, embedding_size=EMB_SIZE, num_classes=nOutputs)
        results = trainer.test(model=model, dataloaders=testDl, verbose=False)
        test_accuracy = results[0]['test/acc']
        accuracies.append(test_accuracy)
        print(f"Attempt {attempt+1}/3 for MLP achieved test accuracy: {test_accuracy:.4f}")
    avgAccuracy = sum(accuracies)/len(accuracies)
    stdAccuracy = (sum([(x-avgAccuracy)**2 for x in accuracies])/len(accuracies))**0.5
    print(f"Final Test Accuracy for MLP: {avgAccuracy:.4f} ± {stdAccuracy:.4f}")
    # Final Test Accuracy for MLP: 0.9516 ± 0.0051


def computeAccuracyMainModelMoreDifficult():
    depth = 7
    memorySize = 100
    lr = 0.001
    useDescendingLR = False
    EMB_SIZE = 256
    TREE_EMB_SIZE = EMB_SIZE
    TREEDECODERLAYERS = 1
    MAXEPOCHS = 50
    batchSize = 32
    maxDigit = 10

    for numDigits in [3, 4, 5]:
        x_train, c_train, y_train = addition_dataset(True, numDigits, maxDigit)
        x_test, c_test, y_test = addition_dataset(False, numDigits, maxDigit)
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
        nConcepts = numDigits * maxDigit
        nOutputs = maxDigit * numDigits - numDigits + 1
        torch.manual_seed(42)
        backBone = MNISTEncoder(emb_size=EMB_SIZE, number_digits=numDigits)
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
        results = trainer.test(model=model, dataloaders=testDl, verbose=False)
        test_accuracy = results[0]['test/task_multiclass_acc']
        print(f"For {numDigits} digits for best config achieved test accuracy: {test_accuracy:.4f}")
        # For 3 digits for best config achieved test accuracy: 0.5752
        # For 4 digits for best config achieved test accuracy: 0.1216
    
def baseModelMNIST():
    #HYPER
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
    #depths = [5]
    memorySizes = [100, 50, 20]
    #memorySizes = [100]
    lrs = [ 0.001]
    descendingLR = [False]
    NUMOFTRIES = 3
    
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
            results = trainer.test(model=model, dataloaders=valDl, verbose=False)
            val_accuracy = results[0]['test/task_multiclass_acc']
            resultsForConfig.append(val_accuracy)
            #print(f"Attempt {attempt+1}/{NUMOFTRIES} for config depth={depth}, memorySize={memorySize}, lr={lr}, useDescendingLR={useDescendingLR} achieved validation accuracy: {val_accuracy:.4f}")
            results2 = trainer.test(model=model, dataloaders=testDl, verbose=False)
            test_accuracy = results2[0]['test/task_multiclass_acc']
            testResultsForConfig.append(test_accuracy)
        testAvgAccuracy = sum(testResultsForConfig)/len(testResultsForConfig)
        testStdAccuracy = (sum([(x-testAvgAccuracy)**2 for x in testResultsForConfig])/len(testResultsForConfig))**0.5
        testAccuracies.append((testAvgAccuracy, testStdAccuracy, (depth, memorySize, lr, useDescendingLR)))
        avgAccuracy = sum(resultsForConfig)/len(resultsForConfig)
        stdAccuracy = (sum([(x-avgAccuracy)**2 for x in resultsForConfig])/len(resultsForConfig))**0.5
        accuracies.append((avgAccuracy, stdAccuracy, (depth, memorySize, lr, useDescendingLR)))
    accuracies.sort(key=lambda x: x[0], reverse=True)
    for acc in accuracies:
        print(f"Accuracy: {acc[0]:.4f} ± {acc[1]:.4f} for config: depth={acc[2][0]}, memorySize={acc[2][1]}, lr={acc[2][2]}, useDescendingLR={acc[2][3]}")
        for testAcc in testAccuracies:
            if testAcc[2] == acc[2]:
                print(f"--> Test Accuracy: {testAcc[0]:.4f} ± {testAcc[1]:.4f} for config: depth={testAcc[2][0]}, memorySize={testAcc[2][1]}, lr={testAcc[2][2]}, useDescendingLR={testAcc[2][3]}")
# NO DROPOUT
#Accuracy: 0.8200 ± 0.0120 for config: depth=5, memorySize=100, lr=0.001, useDescendingLR=False
#--> Test Accuracy: 0.8131 ± 0.0219 for config: depth=5, memorySize=100, lr=0.001, useDescendingLR=False
# Accuracy: 0.7813 ± 0.0111 for config: depth=4, memorySize=50, lr=0.001, useDescendingLR=False
# --> Test Accuracy: 0.7813 ± 0.0111 for config: depth=4, memorySize=50, lr=0.001, useDescendingLR=False
# Accuracy: 0.7790 ± 0.0428 for config: depth=4, memorySize=100, lr=0.001, useDescendingLR=False
# --> Test Accuracy: 0.7790 ± 0.0428 for config: depth=4, memorySize=100, lr=0.001, useDescendingLR=False
# Accuracy: 0.7759 ± 0.0497 for config: depth=5, memorySize=20, lr=0.001, useDescendingLR=False
# --> Test Accuracy: 0.7759 ± 0.0497 for config: depth=5, memorySize=20, lr=0.001, useDescendingLR=False
# Accuracy: 0.7660 ± 0.0488 for config: depth=3, memorySize=100, lr=0.001, useDescendingLR=False
# --> Test Accuracy: 0.7660 ± 0.0488 for config: depth=3, memorySize=100, lr=0.001, useDescendingLR=False
# Accuracy: 0.7493 ± 0.0638 for config: depth=5, memorySize=50, lr=0.001, useDescendingLR=False
# --> Test Accuracy: 0.7493 ± 0.0638 for config: depth=5, memorySize=50, lr=0.001, useDescendingLR=False
# Accuracy: 0.7344 ± 0.0454 for config: depth=3, memorySize=50, lr=0.001, useDescendingLR=False
# --> Test Accuracy: 0.7344 ± 0.0454 for config: depth=3, memorySize=50, lr=0.001, useDescendingLR=False
# Accuracy: 0.6963 ± 0.0431 for config: depth=3, memorySize=20, lr=0.001, useDescendingLR=False
# --> Test Accuracy: 0.6963 ± 0.0431 for config: depth=3, memorySize=20, lr=0.001, useDescendingLR=False
# Accuracy: 0.6448 ± 0.1556 for config: depth=4, memorySize=20, lr=0.001, useDescendingLR=False
# --> Test Accuracy: 0.6448 ± 0.1556 for config: depth=4, memorySize=20, lr=0.001, useDescendingLR=False

#Accuracy: 0.8498 ± 0.0538 for config: depth=5, memorySize=100, lr=0.001, useDescendingLR=True
#--> Test Accuracy: 0.8399 ± 0.0588 for config: depth=5, memorySize=100, lr=0.001, useDescendingLR=True
#Accuracy: 0.8200 ± 0.0120 for config: depth=5, memorySize=100, lr=0.001, useDescendingLR=False
#--> Test Accuracy: 0.8131 ± 0.0219 for config: depth=5, memorySize=100, lr=0.001, useDescendingLR=False

# Accuracy: 0.8524 ± 0.0245 for config: depth=5, memorySize=20, lr=0.001, useDescendingLR=True
# --> Test Accuracy: 0.8537 ± 0.0247 for config: depth=5, memorySize=20, lr=0.001, useDescendingLR=True
# Accuracy: 0.8498 ± 0.0538 for config: depth=5, memorySize=100, lr=0.001, useDescendingLR=True
# --> Test Accuracy: 0.8399 ± 0.0588 for config: depth=5, memorySize=100, lr=0.001, useDescendingLR=True
# Accuracy: 0.8484 ± 0.0607 for config: depth=4, memorySize=100, lr=0.001, useDescendingLR=True
# --> Test Accuracy: 0.8465 ± 0.0607 for config: depth=4, memorySize=100, lr=0.001, useDescendingLR=True
# Accuracy: 0.8149 ± 0.0436 for config: depth=4, memorySize=50, lr=0.001, useDescendingLR=True
# --> Test Accuracy: 0.8068 ± 0.0498 for config: depth=4, memorySize=50, lr=0.001, useDescendingLR=True
# Accuracy: 0.8131 ± 0.1100 for config: depth=5, memorySize=50, lr=0.001, useDescendingLR=True
# --> Test Accuracy: 0.8257 ± 0.1071 for config: depth=5, memorySize=50, lr=0.001, useDescendingLR=True
# Accuracy: 0.8052 ± 0.0893 for config: depth=3, memorySize=100, lr=0.001, useDescendingLR=True
# --> Test Accuracy: 0.8015 ± 0.0945 for config: depth=3, memorySize=100, lr=0.001, useDescendingLR=True
# Accuracy: 0.7291 ± 0.0392 for config: depth=3, memorySize=50, lr=0.001, useDescendingLR=True
# --> Test Accuracy: 0.7297 ± 0.0425 for config: depth=3, memorySize=50, lr=0.001, useDescendingLR=True
# Accuracy: 0.7078 ± 0.1844 for config: depth=4, memorySize=20, lr=0.001, useDescendingLR=True
# --> Test Accuracy: 0.7210 ± 0.1734 for config: depth=4, memorySize=20, lr=0.001, useDescendingLR=True
# Accuracy: 0.7051 ± 0.0491 for config: depth=3, memorySize=20, lr=0.001, useDescendingLR=True
# --> Test Accuracy: 0.7076 ± 0.0539 for config: depth=3, memorySize=20, lr=0.001, useDescendingLR=True


# WITH DROPOUT
# Accuracy: 0.7283 ± 0.0356 for config: depth=5, memorySize=50, lr=0.001, useDescendingLR=False
# --> Test Accuracy: 0.7282 ± 0.0393 for config: depth=5, memorySize=50, lr=0.001, useDescendingLR=False
# Accuracy: 0.7060 ± 0.0477 for config: depth=5, memorySize=100, lr=0.001, useDescendingLR=False
# --> Test Accuracy: 0.7087 ± 0.0404 for config: depth=5, memorySize=100, lr=0.001, useDescendingLR=False
# Accuracy: 0.7003 ± 0.0483 for config: depth=3, memorySize=100, lr=0.001, useDescendingLR=False
# --> Test Accuracy: 0.6990 ± 0.0421 for config: depth=3, memorySize=100, lr=0.001, useDescendingLR=False
# Accuracy: 0.6753 ± 0.1959 for config: depth=4, memorySize=100, lr=0.001, useDescendingLR=False
# --> Test Accuracy: 0.6766 ± 0.1841 for config: depth=4, memorySize=100, lr=0.001, useDescendingLR=False
# Accuracy: 0.6703 ± 0.0873 for config: depth=4, memorySize=50, lr=0.001, useDescendingLR=False
# --> Test Accuracy: 0.6695 ± 0.0858 for config: depth=4, memorySize=50, lr=0.001, useDescendingLR=False
# Accuracy: 0.5968 ± 0.0997 for config: depth=5, memorySize=20, lr=0.001, useDescendingLR=False
# --> Test Accuracy: 0.5992 ± 0.0966 for config: depth=5, memorySize=20, lr=0.001, useDescendingLR=False
# Accuracy: 0.5868 ± 0.0477 for config: depth=3, memorySize=50, lr=0.001, useDescendingLR=False
# --> Test Accuracy: 0.5847 ± 0.0454 for config: depth=3, memorySize=50, lr=0.001, useDescendingLR=False
# Accuracy: 0.5762 ± 0.0339 for config: depth=3, memorySize=20, lr=0.001, useDescendingLR=False
# --> Test Accuracy: 0.5747 ± 0.0260 for config: depth=3, memorySize=20, lr=0.001, useDescendingLR=False
# Accuracy: 0.5439 ± 0.0944 for config: depth=4, memorySize=20, lr=0.001, useDescendingLR=False
# --> Test Accuracy: 0.5565 ± 0.0886 for config: depth=4, memorySize=20, lr=0.001, useDescendingLR=False