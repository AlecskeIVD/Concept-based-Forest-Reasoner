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
from src.CUB.randomF import extractTensorValues

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

class simpleAbstractModel(ConceptMemoryTrees):
    def __init__(self, backbone, embeddingSize, treeEmbeddingSize, treeDecoderNbLayers, nConcepts, nOutputs, batchSize=64, lr=0.001, treeDepth=5, memorySize=100, dropout=False, dropoutP=0.5, preventFullyFalsePaths=False, distribution_weight=0, useDescendingLR=False, useALTLOSS=False, HashedSelector=False, treeLearner="indeptendentFPT", predeterminedTrees = False, givenTrees = None):
        super().__init__(backbone, embeddingSize, treeEmbeddingSize, treeDecoderNbLayers, nConcepts, nOutputs, batchSize=batchSize, lr=lr, treeDepth=treeDepth, memorySize=memorySize, dropout=dropout, dropoutP=dropoutP, preventFullyFalsePaths=preventFullyFalsePaths, distribution_weight=distribution_weight, useDescendingLR=useDescendingLR, useALTLOSS=useALTLOSS, HashedSelector=HashedSelector, treeLearner=treeLearner, predeterminedTrees=predeterminedTrees, givenTrees=givenTrees)
        self.treeSelector = torch.nn.Sequential(
            torch.nn.Linear(embeddingSize, memorySize), # size batchSize, memorySize
        )

        self.conceptPredictor = torch.nn.Sequential(
            torch.nn.Linear(self.embeddingSize, nConcepts),# size batchSize, nConcepts
        )

def extractAccuracyBestMainModelCUB():
    depth = 7
    memorySize = 200
    lr = 0.001/2
    useDescendingLR = False
    EMB_SIZE = 256
    TREE_EMB_SIZE = EMB_SIZE
    TREEDECODERLAYERS = 1
    MAXEPOCHS = 200
    #MAXEPOCHS=1
    nConcepts = len(SELECTED_CONCEPTS)
    nOutputs = 200
    batchSize = 32
    torch.manual_seed(42)
    trainDl, valDl, testDl = createPrecomputedDataloaders(embeddings_file="/Users/alecvandeuren/Thesis/src/data/raw/CUB/CUB_BASE/CUB_200_2011/cub_embeddings.pt", num_random_classes=None, batchSize=batchSize)
    accuracies = []
    for attempt in range(3):
        torch.manual_seed(42+attempt)
        givenTrees = extractTensorValues(depth=depth, memorySize=memorySize, randomState=42+attempt, nConcepts=nConcepts, nOutputs=nOutputs)
        assert givenTrees[0].shape[0] == memorySize and givenTrees[1].shape[0] == memorySize, f"Given trees have incorrect shape: {givenTrees[0].shape}, {givenTrees[1].shape}"
        backBone = LinearEncoder(input_dim=512, output_dim=EMB_SIZE)
        # only 1 phase
        model = simpleAbstractModel(backBone, EMB_SIZE, TREE_EMB_SIZE, TREEDECODERLAYERS, nConcepts, nOutputs, batchSize=batchSize, lr=lr, treeDepth=depth, memorySize=memorySize, dropout=False, dropoutP=0,preventFullyFalsePaths=False, distribution_weight=0, useDescendingLR=useDescendingLR, useALTLOSS=False, HashedSelector=False, treeLearner="predeterminedTrees", predeterminedTrees=True, givenTrees=givenTrees)
        checkpoint_cb = ModelCheckpoint(dirpath="./model/CUB/", save_top_k=1, monitor="val/loss", mode='min')        
        trainer = pl.Trainer(max_epochs=MAXEPOCHS, accelerator='auto', devices="auto", callbacks=[checkpoint_cb, EarlyStopping(monitor="val/loss", mode="min", patience=5)], enable_model_summary=False)
        # if disable validation step, no epoch completed
        trainer.fit(model=model, train_dataloaders=trainDl, val_dataloaders=valDl)
        model = simpleAbstractModel.load_from_checkpoint(checkpoint_cb.best_model_path, backbone=backBone, embeddingSize=EMB_SIZE, treeEmbeddingSize=TREE_EMB_SIZE, treeDecoderNbLayers=TREEDECODERLAYERS, nConcepts=nConcepts, nOutputs=nOutputs, batchSize=batchSize, lr=lr/10, treeDepth=depth, memorySize=memorySize, dropout=False, preventFullyFalsePaths=False, distribution_weight=0, HashedSelector=False, treeLearner="predeterminedTrees", predeterminedTrees=True, givenTrees=givenTrees)
        results = trainer.test(model=model, dataloaders=testDl, verbose=False)
        test_accuracy = results[0]['test/task_multiclass_acc']
        accuracies.append(test_accuracy)
        print(f"Attempt {attempt+1}/3 for best config achieved test accuracy: {test_accuracy:.4f}")
    avgAccuracy = sum(accuracies)/len(accuracies)
    stdAccuracy = (sum([(x-avgAccuracy)**2 for x in accuracies])/len(accuracies))**0.5
    print(f"Final Test Accuracy: {avgAccuracy:.4f} ± {stdAccuracy:.4f} for best config: depth={depth}, memorySize={memorySize}, lr={lr}, useDescendingLR={useDescendingLR}")
    # Final Test Accuracy: 0.3263 ± 0.0143 for best config: depth=7, memorySize=200, lr=0.0005, useDescendingLR=False
