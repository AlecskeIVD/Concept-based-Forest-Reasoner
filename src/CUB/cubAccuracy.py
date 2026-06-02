from functools import reduce
from itertools import product

from src.CUB.cubDataset import createPrecomputedDataloaders, SELECTED_CONCEPTS
from src.CUB.cubEncoder import LinearEncoder
from src.CUB.cubMLP import CUBMLP
import os
import torch
from src.model.abstractModel import ConceptMemoryTrees
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import logging

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

class simpleAbstractModel(ConceptMemoryTrees):
    def __init__(self, backbone, embeddingSize, treeEmbeddingSize, treeDecoderNbLayers, nConcepts, nOutputs, batchSize=64, lr=0.001, treeDepth=5, memorySize=100, dropout=False, dropoutP=0.5, preventFullyFalsePaths=False, distribution_weight=0, useDescendingLR=False, useALTLOSS=False, HashedSelector=False, treeLearner="indeptendentFPT"):
        super().__init__(backbone, embeddingSize, treeEmbeddingSize, treeDecoderNbLayers, nConcepts, nOutputs, batchSize=batchSize, lr=lr, treeDepth=treeDepth, memorySize=memorySize, dropout=dropout, dropoutP=dropoutP, preventFullyFalsePaths=preventFullyFalsePaths, distribution_weight=distribution_weight, useDescendingLR=useDescendingLR, useALTLOSS=useALTLOSS, HashedSelector=HashedSelector, treeLearner=treeLearner)
        self.treeSelector = torch.nn.Sequential(
            torch.nn.Linear(embeddingSize, memorySize), # size batchSize, memorySize
        )

        self.conceptPredictor = torch.nn.Sequential(
            torch.nn.Linear(self.embeddingSize, nConcepts),# size batchSize, nConcepts
        )

def computeAccuracyMainModel():
    #HYPER
    EMB_SIZE = 256
    TREE_EMB_SIZE = EMB_SIZE
    TREEDECODERLAYERS = 1
    nConcepts = len(SELECTED_CONCEPTS)
    nOutputs = 200
    batchSize = 32
    depths = [7, 5, 3]
    memorySizes = [200, 100]
    lrs = [0.001]
    descendingLR = [False]
    MAXEPOCHS = 200
    NUMOFTRIES = 3
    torch.manual_seed(42)
    trainDl, valDl, testDl = createPrecomputedDataloaders(embeddings_file="/Users/alecvandeuren/Thesis/src/data/raw/CUB/CUB_BASE/CUB_200_2011/cub_embeddings.pt", batchSize=batchSize, num_random_classes=None)
    accuracies = []
    for depth, memorySize, lr, useDescendingLR in product(depths, memorySizes, lrs, descendingLR):
        resultsForConfig = []
        for attempt in range(NUMOFTRIES):
            torch.manual_seed(42+attempt)
            backBone = LinearEncoder(input_dim=512, output_dim=EMB_SIZE)
            #PHASE 1: train with dropout max
            model = simpleAbstractModel(backBone, EMB_SIZE, TREE_EMB_SIZE, TREEDECODERLAYERS, nConcepts, nOutputs, batchSize=batchSize, lr=lr, treeDepth=depth, memorySize=memorySize, dropout=True, dropoutP=1,preventFullyFalsePaths=False, distribution_weight=0, useDescendingLR=useDescendingLR, useALTLOSS=False, HashedSelector=True, treeLearner="indeptendentFPT")
            checkpoint_cb = ModelCheckpoint(dirpath="./model/CUB/", save_top_k=1, monitor="val/loss", mode='min')        
            trainer = pl.Trainer(max_epochs=MAXEPOCHS, accelerator='auto', devices="auto", callbacks=[checkpoint_cb, EarlyStopping(monitor="val/loss", mode="min", patience=5)], enable_model_summary=False)
            # if disable validation step, no epoch completed
            trainer.fit(model=model, train_dataloaders=trainDl, val_dataloaders=valDl)

            model = simpleAbstractModel.load_from_checkpoint(checkpoint_cb.best_model_path, backbone=backBone, embeddingSize=EMB_SIZE, treeEmbeddingSize=TREE_EMB_SIZE, treeDecoderNbLayers=TREEDECODERLAYERS, nConcepts=nConcepts, nOutputs=nOutputs, batchSize=batchSize, lr=lr/10, treeDepth=depth, memorySize=memorySize, dropout=False, preventFullyFalsePaths=False, distribution_weight=0, useDescendingLR=False, useALTLOSS=False, HashedSelector=False,  treeLearner="indeptendentFPT")
            model.NoSelector = False

            # PHASE 2: train without dropout, but initialize with the weights of the model trained with dropout
            checkpoint_cb = ModelCheckpoint(dirpath="./model/CUB/", save_top_k=1, monitor="val/loss", mode='min')
            trainer = pl.Trainer(max_epochs=MAXEPOCHS, accelerator='auto', devices="auto", callbacks=[checkpoint_cb, EarlyStopping(monitor="val/loss", mode="min", patience=5)], enable_model_summary=False)
            trainer.fit(model=model, train_dataloaders=trainDl, val_dataloaders=valDl) 
            model = simpleAbstractModel.load_from_checkpoint(checkpoint_cb.best_model_path, backbone=backBone, embeddingSize=EMB_SIZE, treeEmbeddingSize=TREE_EMB_SIZE, treeDecoderNbLayers=TREEDECODERLAYERS, nConcepts=nConcepts, nOutputs=nOutputs, batchSize=batchSize, lr=lr/10, treeDepth=depth, memorySize=memorySize, dropout=False, preventFullyFalsePaths=False, distribution_weight=0, hashedSelector=False, treeLearner="indeptendentFPT")
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
    results = """
    Accuracy: 0.1365 ± 0.0010 for config: depth=7, memorySize=200, lr=0.001, useDescendingLR=False
    Accuracy: 0.1371 ± 0.0024 for config: depth=5, memorySize=200, lr=0.001, useDescendingLR=False
    Accuracy: 0.1243 ± 0.0128 for config: depth=3, memorySize=100, lr=0.001, useDescendingLR=False
Accuracy: 0.1084 ± 0.0163 for config: depth=7, memorySize=100, lr=0.001, useDescendingLR=False
Accuracy: 0.1043 ± 0.0056 for config: depth=5, memorySize=100, lr=0.001, useDescendingLR=False
Accuracy: 0.0801 ± 0.0099 for config: depth=3, memorySize=50, lr=0.001, useDescendingLR=False
Accuracy: 0.0726 ± 0.0165 for config: depth=5, memorySize=50, lr=0.001, useDescendingLR=False
Accuracy: 0.0712 ± 0.0052 for config: depth=7, memorySize=50, lr=0.001, useDescendingLR=False
Accuracy: 0.0609 ± 0.0095 for config: depth=5, memorySize=20, lr=0.001, useDescendingLR=False
Accuracy: 0.0609 ± 0.0034 for config: depth=3, memorySize=20, lr=0.001, useDescendingLR=False
Accuracy: 0.0534 ± 0.0065 for config: depth=7, memorySize=20, lr=0.001, useDescendingLR=False
"""
    bestPerformer = results.split("\n")[0]
    depth = 7
    memorySize = 200
    lr = 0.001
    useDescendingLR = False
    EMB_SIZE = 256
    TREE_EMB_SIZE = EMB_SIZE
    TREEDECODERLAYERS = 1
    MAXEPOCHS = 200
    nConcepts = len(SELECTED_CONCEPTS)
    nOutputs = 200
    batchSize = 32
    torch.manual_seed(42)
    trainDl, valDl, testDl = createPrecomputedDataloaders(embeddings_file="/Users/alecvandeuren/Thesis/src/data/raw/CUB/CUB_BASE/CUB_200_2011/cub_embeddings.pt", num_random_classes=None, batchSize=batchSize)
    accuracies = []
    for attempt in range(3):
        torch.manual_seed(42+attempt)
        backBone = LinearEncoder(input_dim=512, output_dim=EMB_SIZE)
        #PHASE 1: train with dropout max
        model = simpleAbstractModel(backBone, EMB_SIZE, TREE_EMB_SIZE, TREEDECODERLAYERS, nConcepts, nOutputs, batchSize=batchSize, lr=lr, treeDepth=depth, memorySize=memorySize, dropout=True, dropoutP=1,preventFullyFalsePaths=False, distribution_weight=0, useDescendingLR=useDescendingLR, useALTLOSS=False, HashedSelector=True, treeLearner="indeptendentFPT")
        checkpoint_cb = ModelCheckpoint(dirpath="./model/CUB/", save_top_k=1, monitor="val/loss", mode='min')        
        trainer = pl.Trainer(max_epochs=MAXEPOCHS, accelerator='auto', devices="auto", callbacks=[checkpoint_cb, EarlyStopping(monitor="val/loss", mode="min", patience=5)], enable_model_summary=False)
        # if disable validation step, no epoch completed
        trainer.fit(model=model, train_dataloaders=trainDl, val_dataloaders=valDl)
        model = simpleAbstractModel.load_from_checkpoint(checkpoint_cb.best_model_path, backbone=backBone, embeddingSize=EMB_SIZE, treeEmbeddingSize=TREE_EMB_SIZE, treeDecoderNbLayers=TREEDECODERLAYERS, nConcepts=nConcepts, nOutputs=nOutputs, batchSize=batchSize, lr=lr/10, treeDepth=depth, memorySize=memorySize, dropout=False, preventFullyFalsePaths=False, distribution_weight=0, useDescendingLR=False, useALTLOSS=False, HashedSelector=False,  treeLearner="indeptendentFPT")
        model.NoSelector = False
        # PHASE 2: train without dropout, but initialize with the weights of the model trained with dropout
        checkpoint_cb = ModelCheckpoint(dirpath="./model/CUB/", save_top_k=1, monitor="val/loss", mode='min')
        trainer = pl.Trainer(max_epochs=MAXEPOCHS, accelerator='auto', devices="auto", callbacks=[checkpoint_cb, EarlyStopping(monitor="val/loss", mode="min", patience=5)], enable_model_summary=False)
        trainer.fit(model=model, train_dataloaders=trainDl, val_dataloaders=valDl) 
        model = simpleAbstractModel.load_from_checkpoint(checkpoint_cb.best_model_path, backbone=backBone, embeddingSize=EMB_SIZE, treeEmbeddingSize=TREE_EMB_SIZE, treeDecoderNbLayers=TREEDECODERLAYERS, nConcepts=nConcepts, nOutputs=nOutputs, batchSize=batchSize, lr=lr/10, treeDepth=depth, memorySize=memorySize, dropout=False, preventFullyFalsePaths=False, distribution_weight=0, HashedSelector=False, treeLearner="indeptendentFPT")
        results = trainer.test(model=model, dataloaders=testDl, verbose=False)
        test_accuracy = results[0]['test/task_multiclass_acc']
        accuracies.append(test_accuracy)
        print(f"Attempt {attempt+1}/3 for best config achieved test accuracy: {test_accuracy:.4f}")
    avgAccuracy = sum(accuracies)/len(accuracies)
    stdAccuracy = (sum([(x-avgAccuracy)**2 for x in accuracies])/len(accuracies))**0.5
    print(f"Final Test Accuracy: {avgAccuracy:.4f} ± {stdAccuracy:.4f} for best config: depth={depth}, memorySize={memorySize}, lr={lr}, useDescendingLR={useDescendingLR}")
    # Final Test Accuracy: 0.1490 ± 0.0160 for best config: depth=7, memorySize=200, lr=0.001, useDescendingLR=False


def MLPExperiment():
    #EMB_SIZE = 256
    EMB_SIZE = 200
    TREE_EMB_SIZE = EMB_SIZE
    TREEDECODERLAYERS = 1
    MAXEPOCHS = 50
    nConcepts = len(SELECTED_CONCEPTS)
    nOutputs = 200
    batchSize = 32
    torch.manual_seed(42)
    trainDl, valDl, testDl = createPrecomputedDataloaders(embeddings_file="/Users/alecvandeuren/Thesis/src/data/raw/CUB/CUB_BASE/CUB_200_2011/cub_embeddings.pt", batchSize=batchSize)
    accuracies = []
    for attempt in range(3):
        torch.manual_seed(42+attempt)
        encoder = LinearEncoder(input_dim=512, output_dim=EMB_SIZE)
        model = CUBMLP(encoder=encoder, embedding_size=EMB_SIZE, num_classes=nOutputs)
        checkpoint_cb = ModelCheckpoint(dirpath="./model/CUB_MLP/", save_top_k=1, monitor="val/loss", mode='min')        
        trainer = pl.Trainer(max_epochs=MAXEPOCHS, accelerator='auto', devices="auto", callbacks=[checkpoint_cb, EarlyStopping(monitor="val/loss", mode="min", patience=5)], enable_model_summary=False)
        trainer.fit(model=model, train_dataloaders=trainDl, val_dataloaders=valDl)
        model = CUBMLP.load_from_checkpoint(checkpoint_cb.best_model_path, encoder=encoder, embedding_size=EMB_SIZE, num_classes=nOutputs)
        results = trainer.test(model=model, dataloaders=valDl, verbose=False)
        test_accuracy = results[0]['test/acc']
        accuracies.append(test_accuracy)
        print(f"Attempt {attempt+1}/3 for MLP achieved validation accuracy: {test_accuracy:.4f}")
    avgAccuracy = sum(accuracies)/len(accuracies)
    stdAccuracy = (sum([(x-avgAccuracy)**2 for x in accuracies])/len(accuracies))**0.5
    print(f"Final Validation Accuracy for MLP: {avgAccuracy:.4f} ± {stdAccuracy:.4f}")
    # Final Validation Accuracy for MLP: 0.5660 ± 0.0055 WITH ONLY ONE LAYER


def MLPFinalAccuracy():
    #EMB_SIZE = 256
    EMB_SIZE = 200
    TREE_EMB_SIZE = EMB_SIZE
    TREEDECODERLAYERS = 1
    MAXEPOCHS = 50
    nConcepts = len(SELECTED_CONCEPTS)
    nOutputs = 200
    batchSize = 32
    torch.manual_seed(42)
    trainDl, valDl, testDl = createPrecomputedDataloaders(embeddings_file="/Users/alecvandeuren/Thesis/src/data/raw/CUB/CUB_BASE/CUB_200_2011/cub_embeddings.pt", batchSize=batchSize)
    accuracies = []
    for attempt in range(3):
        torch.manual_seed(42+attempt)
        encoder = LinearEncoder(input_dim=512, output_dim=EMB_SIZE)
        model = CUBMLP(encoder=encoder, embedding_size=EMB_SIZE, num_classes=nOutputs)
        checkpoint_cb = ModelCheckpoint(dirpath="./model/CUB_MLP/", save_top_k=1, monitor="val/loss", mode='min')     
        trainer = pl.Trainer(max_epochs=MAXEPOCHS, accelerator='auto', devices="auto", callbacks=[checkpoint_cb, EarlyStopping(monitor="val/loss", mode="min", patience=5)], enable_model_summary=False)
        trainer.fit(model=model, train_dataloaders=trainDl, val_dataloaders=valDl)
        model = CUBMLP.load_from_checkpoint(checkpoint_cb.best_model_path, encoder=encoder, embedding_size=EMB_SIZE, num_classes=nOutputs)
        results = trainer.test(model=model, dataloaders=testDl, verbose=False)
        test_accuracy = results[0]['test/acc']
        accuracies.append(test_accuracy)
        print(f"Attempt {attempt+1}/3 for MLP achieved test accuracy: {test_accuracy:.4f}")
    avgAccuracy = sum(accuracies)/len(accuracies)
    stdAccuracy = (sum([(x-avgAccuracy)**2 for x in accuracies])/len(accuracies))**0.5
    print(f"Final Test Accuracy for MLP: {avgAccuracy:.4f} ± {stdAccuracy:.4f}")
    # Final Test Accuracy for MLP: 0.5658 ± 0.0024



def baseModelCUB():
    #HYPER
    EMB_SIZE = 256
    TREE_EMB_SIZE = EMB_SIZE
    TREEDECODERLAYERS = 1
    nConcepts = len(SELECTED_CONCEPTS)
    nOutputs = 200
    batchSize = 32
    #depths = [7, 5, 3]
    depths=[3]
    #memorySizes = [200, 100]
    memorySizes = [200]
    lrs = [0.001]
    descendingLR = [False]
    MAXEPOCHS = 200
    NUMOFTRIES = 3
    torch.manual_seed(42)
    trainDl, valDl, testDl = createPrecomputedDataloaders(embeddings_file="/Users/alecvandeuren/Thesis/src/data/raw/CUB/CUB_BASE/CUB_200_2011/cub_embeddings.pt", batchSize=batchSize, num_random_classes=None)
    accuracies = []
    testAccuracies = []
    for depth, memorySize, lr, useDescendingLR in product(depths, memorySizes, lrs, descendingLR):
        resultsForConfig = []
        testResultsForConfig = []
        for attempt in range(NUMOFTRIES):
            torch.manual_seed(42+attempt)
            backBone = LinearEncoder(input_dim=512, output_dim=EMB_SIZE)
            #PHASE 1: train with dropout max
            model = simpleAbstractModel(backBone, EMB_SIZE, TREE_EMB_SIZE, TREEDECODERLAYERS, nConcepts, nOutputs, batchSize=batchSize, lr=lr, treeDepth=depth, memorySize=memorySize, dropout=False, dropoutP=0,preventFullyFalsePaths=False, distribution_weight=0, useDescendingLR=useDescendingLR, useALTLOSS=False, HashedSelector=False, treeLearner="indeptendentFPT")
            checkpoint_cb = ModelCheckpoint(dirpath="./model/CUB/", save_top_k=1, monitor="val/loss", mode='min')        
            trainer = pl.Trainer(max_epochs=MAXEPOCHS, accelerator='auto', devices="auto", callbacks=[checkpoint_cb, EarlyStopping(monitor="val/loss", mode="min", patience=5)], enable_model_summary=False)
            # if disable validation step, no epoch completed
            trainer.fit(model=model, train_dataloaders=trainDl, val_dataloaders=valDl)
            model = simpleAbstractModel.load_from_checkpoint(checkpoint_cb.best_model_path, backbone=backBone, embeddingSize=EMB_SIZE, treeEmbeddingSize=TREE_EMB_SIZE, treeDecoderNbLayers=TREEDECODERLAYERS, nConcepts=nConcepts, nOutputs=nOutputs, batchSize=batchSize, lr=lr/10, treeDepth=depth, memorySize=memorySize, dropout=False, preventFullyFalsePaths=False, distribution_weight=0, hashedSelector=False, treeLearner="indeptendentFPT")
            results = trainer.test(model=model, dataloaders=valDl, verbose=False)
            val_accuracy = results[0]['test/task_multiclass_acc']
            resultsForConfig.append(val_accuracy)
            #print(f"Attempt {attempt+1}/{NUMOFTRIES} for config depth={depth}, memorySize={memorySize}, lr={lr}, useDescendingLR={useDescendingLR} achieved validation accuracy: {val_accuracy:.4f}")
            results2 = trainer.test(model=model, dataloaders=testDl, verbose=False)
            test_accuracy = results2[0]['test/task_multiclass_acc']
            testResultsForConfig.append(test_accuracy)
        avgTestAccuracy = sum(testResultsForConfig)/len(testResultsForConfig)
        stdTestAccuracy = (sum([(x-avgTestAccuracy)**2 for x in testResultsForConfig])/len(testResultsForConfig))**0.5
        testAccuracies.append((avgTestAccuracy, stdTestAccuracy, (depth, memorySize, lr, useDescendingLR)))
        avgAccuracy = sum(resultsForConfig)/len(resultsForConfig)
        stdAccuracy = (sum([(x-avgAccuracy)**2 for x in resultsForConfig])/len(resultsForConfig))**0.5
        accuracies.append((avgAccuracy, stdAccuracy, (depth, memorySize, lr, useDescendingLR)))
    accuracies.sort(key=lambda x: x[0], reverse=True)
    for acc in accuracies:
        print(f"Accuracy: {acc[0]:.4f} ± {acc[1]:.4f} for config: depth={acc[2][0]}, memorySize={acc[2][1]}, lr={acc[2][2]}, useDescendingLR={acc[2][3]}")
        depth = acc[2][0]
        memorySize = acc[2][1]
        lr = acc[2][2]
        useDescendingLR = acc[2][3]
        for testAcc in testAccuracies:
            if testAcc[2] == (depth, memorySize, lr, useDescendingLR):
                print(f"Test Accuracy: {testAcc[0]:.4f} ± {testAcc[1]:.4f} for config: depth={testAcc[2][0]}, memorySize={testAcc[2][1]}, lr={testAcc[2][2]}, useDescendingLR={testAcc[2][3]}")
        break
    # NO DROPOUT
    # Accuracy: 0.2049 ± 0.0010 for config: depth=3, memorySize=200, lr=0.001, useDescendingLR=False
    #Test Accuracy: 0.2183 ± 0.0157 for config: depth=3, memorySize=200, lr=0.001, useDescendingLR=False

    # WITH DROPOUT 0.2
#    Accuracy: 0.1929 ± 0.0135 for config: depth=3, memorySize=200, lr=0.001, useDescendingLR=False
#Test Accuracy: 0.2057 ± 0.0103 for config: depth=3, memorySize=200, lr=0.001, useDescendingLR=False



