import math

from src.CMR.cmr import CMR
from functools import reduce
from itertools import product

from src.CUB.cubDataset import SELECTED_CONCEPTS, createPrecomputedDataloaders
from src.CUB.cubEncoder import LinearEncoder
import os
import torch
from src.model.abstractModel import ConceptMemoryTrees
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import logging

def cubCMRExperiment():
    EMB_SIZE = 256
    RULE_EMB_SIZE = EMB_SIZE*4
    TREEDECODERLAYERS = 1
    nConcepts = len(SELECTED_CONCEPTS)
    nOutputs = 200
    batchSize = 64
    #protoWeights = [0, 0.1, 1, 5]
    protoWeights = [0]
    # 200 memory size is too big for GPU
    maxDepthTree = 7
    oldMemorySizes = [200, 100, 50]
    memorySizes = [math.ceil(mem/nOutputs * maxDepthTree) for mem in oldMemorySizes]
    print(f"Using memory sizes: {memorySizes}")
    lrs = [0.001]
    MAXEPOCHS = 20
    NUMOFTRIES = 3
    torch.manual_seed(42)
    trainDl, valDl, testDl = createPrecomputedDataloaders(embeddings_file="/Users/alecvandeuren/Thesis/src/data/raw/CUB/CUB_BASE/CUB_200_2011/cub_embeddings.pt", batchSize=batchSize)
    accuracies = []
    for memSize, lr, pw in product(memorySizes, lrs, protoWeights):
        resultsForConfig = []
        
        for attempt in range(NUMOFTRIES):
            torch.manual_seed(42+attempt)
            backBone = LinearEncoder(input_dim=512, output_dim=EMB_SIZE)
            model = CMR(backbone=backBone, backbone_out_shape=EMB_SIZE, nb_tasks=nOutputs, nb_concepts=nConcepts, nb_rules=memSize, lr=lr, rule_emb_size=RULE_EMB_SIZE, rule_decoder_nb_layers=3, emb_size=EMB_SIZE//2, proto_weight=pw, concept_weight=1)
            checkpoint_callback = ModelCheckpoint(monitor="val/loss", mode="min", filename=f"best-{memSize}-{lr}-{pw}-{{epoch:02d}}-{{val_acc_epoch:.4f}}")
            early_stopping_callback = EarlyStopping(monitor="val/loss", mode="min", patience=3)
            trainer = pl.Trainer(max_epochs=MAXEPOCHS, callbacks=[checkpoint_callback, early_stopping_callback], enable_model_summary=False, accelerator="mps", devices="auto")
            trainer.fit(model, trainDl, valDl)
            best_model_path = checkpoint_callback.best_model_path
            best_model = CMR.load_from_checkpoint(best_model_path, backbone=backBone, backbone_out_shape=EMB_SIZE, nb_tasks=nOutputs, nb_concepts=nConcepts, nb_rules=memSize, lr=lr, rule_emb_size=RULE_EMB_SIZE, rule_decoder_nb_layers=3, emb_size=EMB_SIZE//2, proto_weight=pw, concept_weight=1)
            valAcc = trainer.validate(best_model, valDl)[0]["val/task_multiclass_acc"]
            print(f"iteration {attempt+1}/{NUMOFTRIES} for Memory Size: {memSize}, LR: {lr}, Proto Weight: {pw}, Val Acc: {valAcc}")
            resultsForConfig.append(valAcc)
        avgAcc = sum(resultsForConfig)/len(resultsForConfig)
        stdAcc = (sum([(x-avgAcc)**2 for x in resultsForConfig])/len(resultsForConfig))**0.5
        accuracies.append((avgAcc, stdAcc, (memSize, lr, pw)))
        print(f"Memory Size: {memSize}, LR: {lr}, Proto Weight: {pw}, Avg Val Acc: {avgAcc}, Std Val Acc: {stdAcc}")
    accuracies.sort(key=lambda x: x[0]-x[1], reverse=True)
    for acc in accuracies:
        print(f"Avg Val Acc: {acc[0]}, Std Val Acc: {acc[1]}, Config: {acc[2]}")
"""Avg Val Acc: 0.17681401471296945, Std Val Acc: 0.006706887425554723, Config: (4, 0.001, 0)
Avg Val Acc: 0.12037808944781621, Std Val Acc: 0.018678426667746363, Config: (7, 0.001, 0)
Avg Val Acc: 0.07756463562448819, Std Val Acc: 0.03361730702202135, Config: (2, 0.001, 0)"""

def extractAccuracyCUBCMR():
    EMB_SIZE = 256
    RULE_EMB_SIZE = EMB_SIZE*4
    TREEDECODERLAYERS = 1
    nConcepts = len(SELECTED_CONCEPTS)
    nOutputs = 200
    batchSize = 64
    #protoWeights = [0, 0.1, 1, 5]
    protoWeights = [0]
    # 200 memory size is too big for GPU
    maxDepthTree = 7
    oldMemorySizes = [100]
    memorySizes = [math.ceil(mem/nOutputs * maxDepthTree) for mem in oldMemorySizes]
    print(f"Using memory sizes: {memorySizes}")
    lrs = [0.001]
    MAXEPOCHS = 20
    NUMOFTRIES = 3
    torch.manual_seed(42)
    trainDl, valDl, testDl = createPrecomputedDataloaders(embeddings_file="/Users/alecvandeuren/Thesis/src/data/raw/CUB/CUB_BASE/CUB_200_2011/cub_embeddings.pt", batchSize=batchSize)
    accuracies = []
    for memSize, lr, pw in product(memorySizes, lrs, protoWeights):
        resultsForConfig = []
        
        for attempt in range(NUMOFTRIES):
            torch.manual_seed(42+attempt)
            backBone = LinearEncoder(input_dim=512, output_dim=EMB_SIZE)
            model = CMR(backbone=backBone, backbone_out_shape=EMB_SIZE, nb_tasks=nOutputs, nb_concepts=nConcepts, nb_rules=memSize, lr=lr, rule_emb_size=RULE_EMB_SIZE, rule_decoder_nb_layers=3, emb_size=EMB_SIZE//2, proto_weight=pw, concept_weight=1)
            checkpoint_callback = ModelCheckpoint(monitor="val/loss", mode="min", filename=f"best-{memSize}-{lr}-{pw}-{{epoch:02d}}-{{val_acc_epoch:.4f}}")
            early_stopping_callback = EarlyStopping(monitor="val/loss", mode="min", patience=3)
            trainer = pl.Trainer(max_epochs=MAXEPOCHS, callbacks=[checkpoint_callback, early_stopping_callback], enable_model_summary=False, accelerator="mps", devices="auto")
            trainer.fit(model, trainDl, valDl)
            best_model_path = checkpoint_callback.best_model_path
            best_model = CMR.load_from_checkpoint(best_model_path, backbone=backBone, backbone_out_shape=EMB_SIZE, nb_tasks=nOutputs, nb_concepts=nConcepts, nb_rules=memSize, lr=lr, rule_emb_size=RULE_EMB_SIZE, rule_decoder_nb_layers=3, emb_size=EMB_SIZE//2, proto_weight=pw, concept_weight=1)
            testAcc = trainer.validate(best_model, testDl)[0]["val/task_multiclass_acc"]
            resultsForConfig.append(testAcc)
        avgAcc = sum(resultsForConfig)/len(resultsForConfig)
        stdAcc = (sum([(x-avgAcc)**2 for x in resultsForConfig])/len(resultsForConfig))**0.5
        accuracies.append((avgAcc, stdAcc, (memSize, lr, pw)))
        print(f"CUB Memory Size: {memSize}, LR: {lr}, Proto Weight: {pw}, Avg Val Acc: {avgAcc}, Std Val Acc: {stdAcc}")
        # CUB Memory Size: 4, LR: 0.001, Proto Weight: 0, Avg Val Acc: 0.17282245556513467, Std Val Acc: 0.004421269483716491
