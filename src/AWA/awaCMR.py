import math

from src.CMR.cmr import CMR
from functools import reduce
from itertools import product

from src.AWA.awaDataset import get_awa2_dataloaders, getAWA2EmbeddingDataloaders
from src.AWA.awaEncoder import AWAEncoder, AWALinearBackbone
from src.AWA.awaMLP import AWAMLP
import os
import torch
from src.model.abstractModel import ConceptMemoryTrees
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import logging

def awaCMRExperiment():
    EMB_SIZE = 128
    RULE_EMB_SIZE = EMB_SIZE*4
    TREEDECODERLAYERS = 1
    nConcepts = 85
    nOutputs = 50
    batchSize = 64
    #protoWeights = [0, 0.1, 1, 5]
    protoWeights = [0]
    oldMemorySizes = [100, 50, 20]
    maxDepthTree = 7
    memorySizes = [math.ceil(mem/nOutputs * maxDepthTree) for mem in oldMemorySizes]
    print(f"Using memory sizes: {memorySizes}")
    lrs = [0.001]
    MAXEPOCHS = 50
    NUMOFTRIES = 3
    trainDl, valDl, testDl = getAWA2EmbeddingDataloaders(embeddings_file="/Users/alecvandeuren/Thesis/src/data/raw/AWA/awa_embeddings.pt", root_dir="/Users/alecvandeuren/Thesis/src/data/raw/AWA", batch_size=batchSize)
    accuracies = []
    for memSize, lr, pw in product(memorySizes, lrs, protoWeights):
        resultsForConfig = []
        
        for attempt in range(NUMOFTRIES):
            torch.manual_seed(42+attempt)
            backBone = AWALinearBackbone(input_dim=2048, output_dim=EMB_SIZE)
            model = CMR(backbone=backBone, backbone_out_shape=EMB_SIZE, nb_tasks=nOutputs, nb_concepts=nConcepts, nb_rules=memSize, lr=lr, rule_emb_size=RULE_EMB_SIZE, rule_decoder_nb_layers=3, emb_size=EMB_SIZE//2, proto_weight=pw, concept_weight=1)
            checkpoint_callback = ModelCheckpoint(monitor="val/loss", mode="min", filename=f"best-{memSize}-{lr}-{pw}-{{epoch:02d}}-{{val_acc_epoch:.4f}}")
            early_stopping_callback = EarlyStopping(monitor="val/loss", mode="min", patience=2)
            trainer = pl.Trainer(max_epochs=MAXEPOCHS, callbacks=[checkpoint_callback, early_stopping_callback], enable_model_summary=False, accelerator="mps", devices="auto")
            trainer.fit(model, trainDl, valDl)
            best_model_path = checkpoint_callback.best_model_path
            best_model = CMR.load_from_checkpoint(best_model_path, backbone=backBone, backbone_out_shape=EMB_SIZE, nb_tasks=nOutputs, nb_concepts=nConcepts, nb_rules=memSize, lr=lr, rule_emb_size=RULE_EMB_SIZE, rule_decoder_nb_layers=3, emb_size=EMB_SIZE//2, proto_weight=pw, concept_weight=1)
            valAcc = trainer.validate(best_model, valDl)[0]["val/task_multiclass_acc"]
            print(f"iteration 1/{NUMOFTRIES} for Memory Size: {memSize}, LR: {lr}, Proto Weight: {pw}, Val Acc: {valAcc}")
            resultsForConfig.append(valAcc)
        avgAcc = sum(resultsForConfig)/len(resultsForConfig)
        stdAcc = (sum([(x-avgAcc)**2 for x in resultsForConfig])/len(resultsForConfig))**0.5
        accuracies.append((avgAcc, stdAcc, (memSize, lr, pw)))
        print(f"Memory Size: {memSize}, LR: {lr}, Proto Weight: {pw}, Avg Val Acc: {avgAcc}, Std Val Acc: {stdAcc}")
    accuracies.sort(key=lambda x: x[0]-x[1], reverse=True)
    for acc in accuracies:
        print(f"Avg Val Acc: {acc[0]}, Std Val Acc: {acc[1]}, Config: {acc[2]}")
"""Avg Val Acc: 0.9065639774004618, Std Val Acc: 0.0020416792778273534, Config: (14, 0.001, 0)
Avg Val Acc: 0.9067314068476359, Std Val Acc: 0.00407646891534598, Config: (7, 0.001, 0)
Avg Val Acc: 0.9023777643839518, Std Val Acc: 0.002844982585105201, Config: (3, 0.001, 0)
Avg Val Acc: 0.02829872816801071, Std Val Acc: 0.0, Config: (100, 0.001, 0.1)
Avg Val Acc: 0.02829872816801071, Std Val Acc: 0.0, Config: (100, 0.001, 1)
Avg Val Acc: 0.02829872816801071, Std Val Acc: 0.0, Config: (100, 0.001, 5)
Avg Val Acc: 0.02829872816801071, Std Val Acc: 0.0, Config: (50, 0.001, 0.1)
Avg Val Acc: 0.02829872816801071, Std Val Acc: 0.0, Config: (50, 0.001, 1)
Avg Val Acc: 0.02829872816801071, Std Val Acc: 0.0, Config: (50, 0.001, 5)
Avg Val Acc: 0.02829872816801071, Std Val Acc: 0.0, Config: (20, 0.001, 0.1)
Avg Val Acc: 0.02829872816801071, Std Val Acc: 0.0, Config: (20, 0.001, 1)
Avg Val Acc: 0.02829872816801071, Std Val Acc: 0.0, Config: (20, 0.001, 5)"""


def extractAccuracyAWACMR():
    EMB_SIZE = 128
    RULE_EMB_SIZE = EMB_SIZE*4
    TREEDECODERLAYERS = 1
    nConcepts = 85
    nOutputs = 50
    batchSize = 64
    #protoWeights = [0, 0.1, 1, 5]
    protoWeights = [0]
    oldMemorySizes = [100]
    maxDepthTree = 7
    memorySizes = [math.ceil(mem/nOutputs * maxDepthTree) for mem in oldMemorySizes]
    print(f"Using memory sizes: {memorySizes}")
    lrs = [0.001]
    MAXEPOCHS = 50
    NUMOFTRIES = 3
    trainDl, valDl, testDl = getAWA2EmbeddingDataloaders(embeddings_file="/Users/alecvandeuren/Thesis/src/data/raw/AWA/awa_embeddings.pt", root_dir="/Users/alecvandeuren/Thesis/src/data/raw/AWA", batch_size=batchSize)
    accuracies = []
    for memSize, lr, pw in product(memorySizes, lrs, protoWeights):
        resultsForConfig = []
        
        for attempt in range(NUMOFTRIES):
            torch.manual_seed(42+attempt)
            backBone = AWALinearBackbone(input_dim=2048, output_dim=EMB_SIZE)
            model = CMR(backbone=backBone, backbone_out_shape=EMB_SIZE, nb_tasks=nOutputs, nb_concepts=nConcepts, nb_rules=memSize, lr=lr, rule_emb_size=RULE_EMB_SIZE, rule_decoder_nb_layers=3, emb_size=EMB_SIZE//2, proto_weight=pw, concept_weight=1)
            checkpoint_callback = ModelCheckpoint(monitor="val/loss", mode="min", filename=f"best-{memSize}-{lr}-{pw}-{{epoch:02d}}-{{val_acc_epoch:.4f}}")
            early_stopping_callback = EarlyStopping(monitor="val/loss", mode="min", patience=2)
            trainer = pl.Trainer(max_epochs=MAXEPOCHS, callbacks=[checkpoint_callback, early_stopping_callback], enable_model_summary=False, accelerator="mps", devices="auto")
            trainer.fit(model, trainDl, valDl)
            best_model_path = checkpoint_callback.best_model_path
            best_model = CMR.load_from_checkpoint(best_model_path, backbone=backBone, backbone_out_shape=EMB_SIZE, nb_tasks=nOutputs, nb_concepts=nConcepts, nb_rules=memSize, lr=lr, rule_emb_size=RULE_EMB_SIZE, rule_decoder_nb_layers=3, emb_size=EMB_SIZE//2, proto_weight=pw, concept_weight=1)
            testAcc = trainer.validate(best_model, testDl)[0]["val/task_multiclass_acc"]
            resultsForConfig.append(testAcc)
        avgAcc = sum(resultsForConfig)/len(resultsForConfig)
        stdAcc = (sum([(x-avgAcc)**2 for x in resultsForConfig])/len(resultsForConfig))**0.5
        accuracies.append((avgAcc, stdAcc, (memSize, lr, pw)))
        print(f"AWA DATASET: Memory Size: {memSize}, LR: {lr}, Proto Weight: {pw}, Avg Val Acc: {avgAcc}, Std Val Acc: {stdAcc}")
        # AWA DATASET: Memory Size: 14, LR: 0.001, Proto Weight: 0, Avg Val Acc: 0.9064523379007975, Std Val Acc: 0.0026349031961336307
