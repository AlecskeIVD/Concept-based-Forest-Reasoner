from functools import reduce

from src.CUB.cubDataset import createDataloaders, SELECTED_CONCEPTS, createEmbeddedDataloaders
from src.CUB.cubEncoder import CUBEncoder, LinearEncoder
import os
import torch
from src.model.abstractModel import ConceptMemoryTrees
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class simpleAbstractModel(ConceptMemoryTrees):
    def __init__(self, backbone, embeddingSize, treeEmbeddingSize, treeDecoderNbLayers, nConcepts, nOutputs, batchSize=64, lr=0.001, treeDepth=5, memorySize=100, dropout=False, dropoutP=0.5, preventFullyFalsePaths=False, distribution_weight=0, useDescendingLR=False, useALTLOSS=False, HashedSelector=False, treeLearner="indeptendentFPT"):
        super().__init__(backbone, embeddingSize, treeEmbeddingSize, treeDecoderNbLayers, nConcepts, nOutputs, batchSize=batchSize, lr=lr, treeDepth=treeDepth, memorySize=memorySize, dropout=dropout, dropoutP=dropoutP, preventFullyFalsePaths=preventFullyFalsePaths, distribution_weight=distribution_weight, useDescendingLR=useDescendingLR, useALTLOSS=useALTLOSS, HashedSelector=HashedSelector, treeLearner=treeLearner)
        self.treeSelector = torch.nn.Sequential(
            torch.nn.Linear(embeddingSize, memorySize), # size batchSize, memorySize
        )

        self.conceptPredictor = torch.nn.Sequential(
            torch.nn.Linear(self.embeddingSize, nConcepts),# size batchSize, nConcepts
        )
    

def runExperiment():

    #HYPER
    EMB_SIZE = 256
    TREE_EMB_SIZE = EMB_SIZE
    TREEDECODERLAYERS = 3
    nConcepts = len(SELECTED_CONCEPTS)
    nOutputs = 200
    batchSize = 64
    depth = 8
    memorySize = 100 
    lr = 0.005
    MAXEPOCHS = 100
    
    embedder = CUBEncoder()
    backBone = LinearEncoder(input_dim=embedder.output_dim, output_dim=EMB_SIZE)


    trainDl, valDL = createEmbeddedDataloaders(batchSize)
    #PHASE 1: train with dropout max
    hists = []
    #model = ConceptMemoryTrees(backBone, EMB_SIZE, TREE_EMB_SIZE, TREEDECODERLAYERS, nConcepts, nOutputs, batchSize=batchSize, lr=lr, treeDepth=depth, memorySize=memorySize, dropout=True, dropoutP=1,preventFullyFalsePaths=False, distribution_weight=0, useDescendingLR=True, useALTLOSS=False, HashedSelector=True, treeLearner="indeptendentFPT")
    model = simpleAbstractModel(backBone, EMB_SIZE, TREE_EMB_SIZE, TREEDECODERLAYERS, nConcepts, nOutputs, batchSize=batchSize, lr=lr, treeDepth=depth, memorySize=memorySize, dropout=True, dropoutP=1,preventFullyFalsePaths=False, distribution_weight=0, useDescendingLR=True, useALTLOSS=False, HashedSelector=True, treeLearner="indeptendentFPT")
    checkpoint_cb = ModelCheckpoint(dirpath="./model/CUB/", save_top_k=1, monitor="val/loss", mode='min')        
    checkpoint_cb = ModelCheckpoint(dirpath="./model/CUB/", save_top_k=1, monitor="val/loss", mode='min')        
    trainer = pl.Trainer(max_epochs=MAXEPOCHS, accelerator='auto', devices="auto", callbacks=[checkpoint_cb, EarlyStopping(monitor="val/loss", mode="min", patience=100)])
    # if disable validation step, no epoch completed
    trainer.fit(model=model, train_dataloaders=trainDl, val_dataloaders=valDL)       
    combined_tensor = reduce(torch.logical_or, model.maskMap.values())

    print(combined_tensor)
    try:
        model.printTrees()
    except Exception as e:
        print("Error printing trees:", e)
    model = simpleAbstractModel.load_from_checkpoint(checkpoint_cb.best_model_path, backbone=backBone, embeddingSize=EMB_SIZE, treeEmbeddingSize=TREE_EMB_SIZE, treeDecoderNbLayers=TREEDECODERLAYERS, nConcepts=nConcepts, nOutputs=nOutputs, batchSize=batchSize, lr=lr, treeDepth=depth, memorySize=memorySize, dropout=False, preventFullyFalsePaths=False, distribution_weight=0, useDescendingLR=False, useALTLOSS=False, HashedSelector=False,  treeLearner="indeptendentFPT")
    model.NoSelector = False

    # PHASE 2: train without dropout, but initialize with the weights of the model trained with dropout
    checkpoint_cb = ModelCheckpoint(dirpath="./model/CUB/", save_top_k=1, monitor="val/loss", mode='min')
    trainer = pl.Trainer(max_epochs=MAXEPOCHS*10, accelerator='auto', devices="auto", callbacks=[checkpoint_cb, EarlyStopping(monitor="val/loss", mode="min", patience=100)])
    trainer.fit(model=model, train_dataloaders=trainDl, val_dataloaders=valDL) 
    model = simpleAbstractModel.load_from_checkpoint(checkpoint_cb.best_model_path, backbone=backBone, embeddingSize=EMB_SIZE, treeEmbeddingSize=TREE_EMB_SIZE, treeDecoderNbLayers=TREEDECODERLAYERS, nConcepts=nConcepts, nOutputs=nOutputs, batchSize=batchSize, lr=0.001, treeDepth=depth, memorySize=memorySize, dropout=False, preventFullyFalsePaths=False, distribution_weight=0, hashedSelector=False, treeLearner="indeptendentFPT")
    try:
        model.printTrees()
    except Exception as e:
        print("Error printing trees:", e)

    # PRETRAINDE FEATURE EXTRACTOR
    # werkt gewoon neuraal netwerk?
    # GEBRUIK ECHTE CUB IPV DEZE FOTOSHOP SHIT
    # 2 phase training
    # Batch entropy?
    # Better initialization?
    # First phase: Use zero'd selector, but still let it train
    # Hyper neural networks
    # BEGIN MET LINEAR LAYER NA PRETRAINED MODEL OM OVERFITTING TE VERMIJDEN


def MLPExperimentBAD():
    from src.CUB.cubMLP import CUBMLP
    trainDl, valDl = createEmbeddedDataloaders(batchSize=64)
    embedder = CUBEncoder()
    model = CUBMLP(input_dim=embedder.output_dim, hidden_dim=128, output_dim=200, lr=0.001)
    checkpoint_cb = ModelCheckpoint(dirpath="./model/CUB/MLP/", save_top_k=1, monitor="val/loss", mode='min')
    trainer = pl.Trainer(max_epochs=100, accelerator='auto', devices="auto", callbacks=[checkpoint_cb, EarlyStopping(monitor="val/loss", mode="min", patience=100)])
    trainer.fit(model=model, train_dataloaders=trainDl, val_dataloaders=valDl)

def MLPExperiment():
    from src.CUB.cubMLP import fullCUBMLP
    trainDl, valDl = createDataloaders(batchSize=32)
    model = fullCUBMLP(200)
    checkpoint_cb = ModelCheckpoint(dirpath="./model/CUB/MLP/", save_top_k=1, monitor="val/loss", mode='min')
    trainer = pl.Trainer(max_epochs=100, accelerator='auto', devices="auto", callbacks=[checkpoint_cb, EarlyStopping(monitor="val/loss", mode="min", patience=100)])
    trainer.fit(model=model, train_dataloaders=trainDl, val_dataloaders=valDl)
