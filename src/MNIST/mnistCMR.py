import math

from src.CMR.cmr import CMR
from functools import reduce
from itertools import product

from src.MNIST.mnist_dataset import addition_dataset
from src.MNIST.mnistEncoder import MNISTEncoder
import os
import torch
from src.model.abstractModel import ConceptMemoryTrees
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import logging

def mnistCMRExperiment():
    EMB_SIZE = 128
    RULE_EMB_SIZE = EMB_SIZE*4
    TREEDECODERLAYERS = 1
    nConcepts = 20
    nOutputs = 19
    batchSize = 64
    #protoWeights = [0, 0.1, 1, 5]
    maxDepthTree = 5
    protoWeights = [0]
    oldMemorySizes = [100, 50, 20]
    memorySizes = [math.ceil(mem/nOutputs * maxDepthTree) for mem in oldMemorySizes]
    print(f"Using memory sizes: {memorySizes}")
    lrs = [0.001]
    MAXEPOCHS = 50
    NUMOFTRIES = 3
    torch.manual_seed(42)
    x_train, c_train, y_train = addition_dataset(True, 2, 10)
    x_test, c_test, y_test = addition_dataset(False, 2, 10)
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
    for memSize, lr, pw in product(memorySizes, lrs, protoWeights):
        resultsForConfig = []
        
        for attempt in range(NUMOFTRIES):
            torch.manual_seed(42+attempt)
            backBone = MNISTEncoder(number_digits=2, emb_size=EMB_SIZE)
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
"""Avg Val Acc: 0.9441111286481222, Std Val Acc: 0.007161278037739869, Config: (14, 0.001, 0)
Avg Val Acc: 0.9330000082651774, Std Val Acc: 0.00641757598093405, Config: (27, 0.001, 0)
Avg Val Acc: 0.9096666773160299, Std Val Acc: 0.027716810437411383, Config: (6, 0.001, 0)
"""
def extractAccuracyMNISTCMR():
    EMB_SIZE = 128
    RULE_EMB_SIZE = EMB_SIZE*4
    TREEDECODERLAYERS = 1
    nConcepts = 20
    nOutputs = 19
    batchSize = 64
    #protoWeights = [0, 0.1, 1, 5]
    maxDepthTree = 5
    protoWeights = [0]
    oldMemorySizes = [50]
    memorySizes = [math.ceil(mem/nOutputs * maxDepthTree) for mem in oldMemorySizes]
    print(f"Using memory sizes: {memorySizes}")
    lrs = [0.001]
    MAXEPOCHS = 50
    NUMOFTRIES = 3
    torch.manual_seed(42)
    x_train, c_train, y_train = addition_dataset(True, 2, 10)
    x_test, c_test, y_test = addition_dataset(False, 2, 10)
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
    for memSize, lr, pw in product(memorySizes, lrs, protoWeights):
        resultsForConfig = []
        
        for attempt in range(NUMOFTRIES):
            torch.manual_seed(42+attempt)
            backBone = MNISTEncoder(number_digits=2, emb_size=EMB_SIZE)
            model = CMR(backbone=backBone, backbone_out_shape=EMB_SIZE, nb_tasks=nOutputs, nb_concepts=nConcepts, nb_rules=memSize, lr=lr, rule_emb_size=RULE_EMB_SIZE, rule_decoder_nb_layers=3, emb_size=EMB_SIZE//2, proto_weight=pw, concept_weight=1)
            checkpoint_callback = ModelCheckpoint(monitor="val/loss", mode="min", filename=f"best-{memSize}-{lr}-{pw}-{{epoch:02d}}-{{val_acc_epoch:.4f}}")
            early_stopping_callback = EarlyStopping(monitor="val/loss", mode="min", patience=2)
            trainer = pl.Trainer(max_epochs=MAXEPOCHS, callbacks=[checkpoint_callback, early_stopping_callback], enable_model_summary=False, accelerator="mps", devices="auto")
            trainer.fit(model, trainDl, valDl)
            best_model_path = checkpoint_callback.best_model_path
            best_model = CMR.load_from_checkpoint(best_model_path, backbone=backBone, backbone_out_shape=EMB_SIZE, nb_tasks=nOutputs, nb_concepts=nConcepts, nb_rules=memSize, lr=lr, rule_emb_size=RULE_EMB_SIZE, rule_decoder_nb_layers=3, emb_size=EMB_SIZE//2, proto_weight=pw, concept_weight=1)
            valAcc = trainer.validate(best_model, testDl)[0]["val/task_multiclass_acc"]
            resultsForConfig.append(valAcc)
        avgAcc = sum(resultsForConfig)/len(resultsForConfig)
        stdAcc = (sum([(x-avgAcc)**2 for x in resultsForConfig])/len(resultsForConfig))**0.5
        accuracies.append((avgAcc, stdAcc, (memSize, lr, pw)))
        print(f"MNIST Memory Size: {memSize}, LR: {lr}, Proto Weight: {pw}, Avg Val Acc: {avgAcc}, Std Val Acc: {stdAcc}")
        # MNIST Memory Size: 14, LR: 0.001, Proto Weight: 0, Avg Val Acc: 0.9404000043869019, Std Val Acc: 0.009896816590207191


def computeAccuracyMNISTMoreDifficult():
    EMB_SIZE = 128
    RULE_EMB_SIZE = EMB_SIZE*4
    TREEDECODERLAYERS = 1
    nConcepts = 20
    nOutputs = 19
    batchSize = 64
    #protoWeights = [0, 0.1, 1, 5]
    maxDepthTree = 5
    protoWeights = [0]
    oldMemorySizes = [50]
    memorySizes = [math.ceil(mem/nOutputs * maxDepthTree) for mem in oldMemorySizes]
    memorySize = max(memorySizes)
    protoWeight = protoWeights[0]
    lr = 0.001

    print(f"Using memory sizes: {memorySizes}")
    lrs = [0.001]
    MAXEPOCHS = 50
    EMB_SIZE = 256
    for numDigits in [3, 4, 5]:
        x_train, c_train, y_train = addition_dataset(True, numDigits, 10)
        x_test, c_test, y_test = addition_dataset(False, numDigits, 10)
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
        nConcepts = numDigits * 10
        nOutputs = 10 * numDigits - numDigits + 1
        torch.manual_seed(42)
        backBone = MNISTEncoder(number_digits=numDigits, emb_size=EMB_SIZE)
        model = CMR(backbone=backBone, backbone_out_shape=EMB_SIZE, nb_tasks=nOutputs, nb_concepts=nConcepts, nb_rules=memorySize, lr=lr, rule_emb_size=RULE_EMB_SIZE, rule_decoder_nb_layers=3, emb_size=EMB_SIZE//2, proto_weight=protoWeight, concept_weight=1)
        checkpoint_callback = ModelCheckpoint(monitor="val/loss", mode="min", filename=f"best-{memorySize}-{lr}-{protoWeight}-{{epoch:02d}}-{{val_acc_epoch:.4f}}")
        early_stopping_callback = EarlyStopping(monitor="val/loss", mode="min", patience=2)
        trainer = pl.Trainer(max_epochs=MAXEPOCHS, callbacks=[checkpoint_callback, early_stopping_callback], enable_model_summary=False, accelerator="mps", devices="auto")
        trainer.fit(model, trainDl, valDl)
        best_model_path = checkpoint_callback.best_model_path
        best_model = CMR.load_from_checkpoint(best_model_path, backbone=backBone, backbone_out_shape=EMB_SIZE, nb_tasks=nOutputs, nb_concepts=nConcepts, nb_rules=memorySize, lr=lr, rule_emb_size=RULE_EMB_SIZE, rule_decoder_nb_layers=3, emb_size=EMB_SIZE//2, proto_weight=protoWeight, concept_weight=1)
        valAcc = trainer.validate(best_model, testDl)[0]["val/task_multiclass_acc"]
        print(f"For {numDigits} digits for best config achieved test accuracy: {valAcc:.4f}")
        # For 3 digits for best config achieved test accuracy: 0.7369
        # For 4 digits for best config achieved test accuracy: 0
    
