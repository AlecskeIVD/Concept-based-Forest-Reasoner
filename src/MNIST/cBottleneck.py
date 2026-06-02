from matplotlib import pyplot as plt

from src.MNIST.mnist_dataset import addition_dataset
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
import xgboost as xgb
from sklearn.metrics import accuracy_score
from src.MNIST.mnistXGB import extract_numpy_data, mnistConceptPredictor


def bottleNeckExperiment():
    plotVals = []
    nConcepts = [5, 10, 15, 20]
    depth = 5
    memorySize = 50
    lr = 0.001/2
    useDescendingLR = False
    EMB_SIZE = 128
    TREE_EMB_SIZE = EMB_SIZE
    TREEDECODERLAYERS = 1
    MAXEPOCHS = 100
    #nConcepts = 20
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

    for nConcept in nConcepts:
        c_train_subset = c_train[:, :nConcept]
        c_val_subset = c_val[:, :nConcept]
        c_test_subset = c_test[:, :nConcept]
        trainDl = DataLoader(TensorDataset(x_train, c_train_subset, y_train), batch_size=batchSize, shuffle=False, num_workers=7, persistent_workers=True)
        testDl = DataLoader(TensorDataset(x_test, c_test_subset, y_test), batch_size=batchSize,  num_workers=7, persistent_workers=True)
        valDl = DataLoader(TensorDataset(x_val, c_val_subset, y_val), batch_size=batchSize,  num_workers=7, persistent_workers=True)
        backBone = MNISTEncoder(emb_size=128, number_digits=2)
        #PHASE 1: train with dropout max
        model = ConceptMemoryTrees(backBone, EMB_SIZE, TREE_EMB_SIZE, TREEDECODERLAYERS, nConcept, nOutputs, batchSize=batchSize, lr=lr, treeDepth=depth, memorySize=memorySize, dropout=True, dropoutP=1,preventFullyFalsePaths=False, distribution_weight=0, useDescendingLR=useDescendingLR, useALTLOSS=False, HashedSelector=True, treeLearner="indeptendentFPT")
        checkpoint_cb = ModelCheckpoint(dirpath="./model/MNIST/", save_top_k=1, monitor="val/loss", mode='min')        
        trainer = pl.Trainer(max_epochs=MAXEPOCHS, accelerator='auto', devices="auto", callbacks=[checkpoint_cb, EarlyStopping(monitor="val/loss", mode="min", patience=5)], enable_model_summary=False)
        # if disable validation step, no epoch completed
        trainer.fit(model=model, train_dataloaders=trainDl, val_dataloaders=valDl)
        model = ConceptMemoryTrees.load_from_checkpoint(checkpoint_cb.best_model_path, backbone=backBone, embeddingSize=EMB_SIZE, treeEmbeddingSize=TREE_EMB_SIZE, treeDecoderNbLayers=TREEDECODERLAYERS, nConcepts=nConcept, nOutputs=nOutputs, batchSize=batchSize, lr=lr/10, treeDepth=depth, memorySize=memorySize, dropout=False, preventFullyFalsePaths=False, distribution_weight=0, useDescendingLR=False, useALTLOSS=False, HashedSelector=False,  treeLearner="indeptendentFPT")
        model.NoSelector = False
        # PHASE 2: train without dropout, but initialize with the weights of the model trained with dropout
        checkpoint_cb = ModelCheckpoint(dirpath="./model/MNIST/", save_top_k=1, monitor="val/loss", mode='min')
        trainer = pl.Trainer(max_epochs=MAXEPOCHS, accelerator='auto', devices="auto", callbacks=[checkpoint_cb, EarlyStopping(monitor="val/loss", mode="min", patience=5)], enable_model_summary=False)
        trainer.fit(model=model, train_dataloaders=trainDl, val_dataloaders=valDl) 
        model = ConceptMemoryTrees.load_from_checkpoint(checkpoint_cb.best_model_path, backbone=backBone, embeddingSize=EMB_SIZE, treeEmbeddingSize=TREE_EMB_SIZE, treeDecoderNbLayers=TREEDECODERLAYERS, nConcepts=nConcept, nOutputs=nOutputs, batchSize=batchSize, lr=lr/10, treeDepth=depth, memorySize=memorySize, dropout=False, preventFullyFalsePaths=False, distribution_weight=0, hashedSelector=False, treeLearner="indeptendentFPT")
        results = trainer.test(model=model, dataloaders=testDl, verbose=False)
        test_accuracy = results[0]['test/task_multiclass_acc']
        plotVals.append((nConcept, test_accuracy))
        print(f"nConcepts: {nConcept}, test accuracy: {test_accuracy:.4f}")
    
    plt.plot([r[0] for r in plotVals], [r[1] for r in plotVals], marker='o')
    plt.title("Test Accuracy vs Number of Concepts")
    plt.xlabel("Number of Concepts")
    plt.ylabel("Test Accuracy")
    plt.xticks([5, 10, 15, 20])
    plt.grid()
    plt.savefig("bottleneck_experiment.png")


def bottleNeckExperiment2():
    # train CBM with XGB
    resultsTwoStage = [(5, 0.4014), (10, 0.4066), (15, 0.7774), (20, 0.9348)]
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
    resultsXGB = []
    for depth, mem in product(depths, memorySizes):
        for nConcept in [5, 10, 15, 20]:
            torch.manual_seed(42)
            c_train_subset = c_train[:, :nConcept]
            c_val_subset = c_val[:, :nConcept]
            c_test_subset = c_test[:, :nConcept]
            trainDl = DataLoader(TensorDataset(x_train, c_train_subset, y_train), batch_size=batchSize, shuffle=False, num_workers=7, persistent_workers=True)
            testDl = DataLoader(TensorDataset(x_test, c_test_subset, y_test), batch_size=batchSize,  num_workers=7, persistent_workers=True)
            valDl = DataLoader(TensorDataset(x_val, c_val_subset, y_val), batch_size=batchSize,  num_workers=7, persistent_workers=True)
            X_train, Y_train = extract_numpy_data(trainDl)

            X_val, Y_val = extract_numpy_data(valDl)

            X_test, Y_test = extract_numpy_data(testDl)
            xgb_model = xgb.XGBClassifier(
                n_estimators=mem,      
                max_depth=depth,           
                learning_rate=0.1,     
                objective='multi:softmax', 
                tree_method='hist',    
                n_jobs=-1,
                early_stopping_rounds=15,
                random_state=42
            )

            xgb_model.fit(
                X_train, Y_train,
                eval_set=[(X_val, Y_val)], 
                verbose=10                 
            )
            encoder = MNISTEncoder(EMB_SIZE, number_digits=2)
            conceptPredictor = mnistConceptPredictor(encoder=encoder, embedding_size=EMB_SIZE, num_classes=nConcept)
            checkpoint_cb = ModelCheckpoint(dirpath="./model/CUB/", save_top_k=1, monitor="val/loss", mode='min')        
            trainer = pl.Trainer(max_epochs=maxEpochs, accelerator='auto', devices="auto", callbacks=[checkpoint_cb, EarlyStopping(monitor="val/loss", mode="min", patience=5)], enable_model_summary=False)
            trainer.fit(model=conceptPredictor, train_dataloaders=trainDl, val_dataloaders=valDl)
            conceptPredictor = mnistConceptPredictor.load_from_checkpoint(checkpoint_cb.best_model_path, encoder=encoder, embedding_size=EMB_SIZE, num_classes=nConcept)
            cPreds = conceptPredictor.extractConceptPreds(testDl)
            print(f"Extracted concept predictions shape: {cPreds.shape}")
            yPreds = xgb_model.predict(cPreds)
            acc = accuracy_score(Y_test, yPreds)
            resultsXGB.append((nConcept, acc))
            print(f"XGB - nConcepts: {nConcept}, test accuracy: {acc:.4f}")
        
    plt.plot([r[0] for r in resultsTwoStage], [r[1] for r in resultsTwoStage], marker='o', label="Two Stage Training")
    plt.plot([r[0] for r in resultsXGB], [r[1] for r in resultsXGB], marker='o', label="XGB with Concept Predictions")
    plt.title("Test Accuracy vs Number of Concepts")
    plt.xlabel("Number of Concepts")
    plt.ylabel("Test Accuracy")
    plt.xticks([5, 10, 15, 20])
    plt.grid()
    plt.legend()
    plt.savefig("bottleneck_experiment_comparison.png")
