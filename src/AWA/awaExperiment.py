from functools import reduce

from src.AWA.awaDataset import get_awa2_dataloaders, getAWA2EmbeddingDataloaders
from src.AWA.awaEncoder import AWAEncoder, AWALinearBackbone
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
    
    def printTrees(self, *args):
        treeVarsProbs_tcv, leafValues_tvo = self.embeddingTranslator(self.dtEmbeddings.weight) # nb_trees, nb_variablesTree
        treeVarsProbs_tcv = treeVarsProbs_tcv.detach().cpu()
        leafValues_tvo = leafValues_tvo.detach().cpu()
        for tree in range(self.memorySize):
            print(f"Tree {tree}:")
            treeVars = treeVarsProbs_tcv[tree,: , :]
            splitVars = treeVars
            leafValues = leafValues_tvo[tree,:, :].view(2**self.treeDepth, self.nClasses)

            indexToClass = {
                1-1:"antelope",
                 2-1:"grizzly+bear",
                 3-1:"killer+whale",
                 4-1:"beaver",
                 5-1:"dalmatian",
                 6-1:"persian",
                 7-1:"horse",
                 8-1:"german+shepherd",
                 9-1:"blue+whale",
                10-1:"siamese+cat",
                11-1:"skunk",
                12-1:"mole",
                13-1:"tiger",
                14-1:"hippopotamus",
                15-1:"leopard",
                16-1:"moose",
                17-1:"spider+monkey",
                18-1:"humpbac+whale",
                19-1:"elephant",
                20-1:"gorilla",
                21-1:"ox",
                22-1:"fox",
                23-1:"sheep",
                24-1:"seal",
                25-1:"chimpan",
                26-1:"hamster",
                27-1:"squirre",
                28-1:"rhinocerus",
                29-1:"rabbit",
                30-1:"bat",
                31-1:"giraffe",
                32-1:"wolf",
                33-1:"chihuauah",
                34-1:"rat",
                35-1:"weasel",
                36-1:"otter",
                37-1:"buffalo",
                38-1:"zebra",
                39-1:"giant+panda",
                40-1:"deer",
                41-1:"bobcat",
                42-1:"pig",
                43-1:"lion",
                44-1:"mouse",
                45-1:"polar+bear",
                46-1:"collie",
                47-1:"walrus",
                48-1:"raccoon",
                49-1:"cow",
                50-1:"dolphin",
            }
            attributes = {
                 1-1: "black",                 2-1: "white",                 3-1: "blue",                 4-1: "brown",                 5-1: "gray",                 6-1: "orange",                 7-1: "red",                 8-1: "yellow",                 9-1: "patches",                10-1: "spots",                11-1: "stripes",                12-1: "furry",                13-1: "hairless",                14-1: "toughskin",                15-1: "big",                16-1: "small",                17-1: "bulbous",                18-1: "lean",                19-1: "flippers",                20-1: "hands",                21-1: "hooves",                22-1: "pads",                23-1: "paws",                24-1: "longleg",                25-1: "longneck",                26-1: "tail",                27-1: "chewteeth",                28-1: "meatteeth",                29-1: "buckteeth",                30-1: "strainteeth",                31-1: "horns",                32-1: "claws",                33-1: "tusks",                34-1: "smelly",                35-1: "flys",                36-1: "hops",                37-1: "swims",                38-1: "tunnels",                39-1: "walks",                40-1: "fast",                41-1: "slow",                42-1: "strong",                43-1: "weak",                44-1: "muscle",                45-1: "bipedal",                46-1: "quadrapedal",                47-1: "active",                48-1: "inactive",                49-1: "nocturnal",                50-1: "hibernate",                51-1: "agility",                52-1: "fish",                53-1: "meat",                54-1: "plankton",                55-1: "vegetation",                56-1: "insects",                57-1: "forager",                58-1: "grazer",                59-1: "hunter",                60-1: "scavenger",                61-1: "skimmer",                62-1: "stalker",                63-1: "newworld",                64-1: "oldworld",                65-1: "arctic",                66-1: "coastal",                67-1: "desert",                68-1: "bush",                69-1: "plains",                70-1: "forest",                71-1: "fields",                72-1: "jungle",                73-1: "mountains",                74-1: "ocean",                75-1: "ground",                76-1: "water",                77-1: "tree",                78-1: "cave",                79-1: "fierce",                80-1: "timid",                81-1: "smart",                82-1: "group",                83-1: "solitary",                84-1: "nestspot",                85-1: "domestic",            
            }
            
            for depth in range(self.treeDepth):
                numInterNodes = 2**depth
                base = 2**depth-1
                for node in range(numInterNodes):
                    splitVar = torch.argmax(splitVars[:,base+node]).item()
                    print(f"{attributes[splitVar]}",end=' | ')
                print()
            for leaf in range(2**self.treeDepth):
                classIndex = torch.argmax(leafValues[leaf, :]).item()
                print(f"{indexToClass[classIndex]}", end=' | ')
            print()
            print("---------")

def runExperiment():

    #HYPER
    EMB_SIZE = 256
    TREE_EMB_SIZE = EMB_SIZE
    TREEDECODERLAYERS = 3
    nConcepts = 85
    nOutputs = 50
    batchSize = 64
    depth = 5
    memorySize = 100 
    lr = 0.001
    MAXEPOCHS = 10
    
    backBone = AWAEncoder(output_dim=EMB_SIZE)


    trainDl, valDl, testDl = get_awa2_dataloaders(root_dir="/Users/alecvandeuren/Thesis/src/data/raw/AWA", batch_size=batchSize)
    #PHASE 1: train with dropout max
    hists = []
    model = ConceptMemoryTrees(backBone, EMB_SIZE, TREE_EMB_SIZE, TREEDECODERLAYERS, nConcepts, nOutputs, batchSize=batchSize, lr=lr, treeDepth=depth, memorySize=memorySize, dropout=True, dropoutP=1,preventFullyFalsePaths=False, distribution_weight=0, useDescendingLR=True, useALTLOSS=False, HashedSelector=True, treeLearner="indeptendentFPT")
    checkpoint_cb = ModelCheckpoint(dirpath="./model/AWA/", save_top_k=1, monitor="val/loss", mode='min')        
    checkpoint_cb = ModelCheckpoint(dirpath="./model/AWA/", save_top_k=1, monitor="val/loss", mode='min')        
    trainer = pl.Trainer(max_epochs=MAXEPOCHS, accelerator='auto', devices="auto", callbacks=[checkpoint_cb, EarlyStopping(monitor="val/loss", mode="min", patience=2)])
    # if disable validation step, no epoch completed
    trainer.fit(model=model, train_dataloaders=trainDl, val_dataloaders=valDl)       
    combined_tensor = reduce(torch.logical_or, model.maskMap.values())

    print(combined_tensor)
    try:
        model.printTrees()
    except Exception as e:
        print("Error printing trees:", e)
    model = ConceptMemoryTrees.load_from_checkpoint(checkpoint_cb.best_model_path, backbone=backBone, embeddingSize=EMB_SIZE, treeEmbeddingSize=TREE_EMB_SIZE, treeDecoderNbLayers=TREEDECODERLAYERS, nConcepts=nConcepts, nOutputs=nOutputs, batchSize=batchSize, lr=lr, treeDepth=depth, memorySize=memorySize, dropout=False, preventFullyFalsePaths=False, distribution_weight=0, useDescendingLR=False, useALTLOSS=False, HashedSelector=False,  treeLearner="indeptendentFPT")
    model.NoSelector = False

    # PHASE 2: train without dropout, but initialize with the weights of the model trained with dropout
    checkpoint_cb = ModelCheckpoint(dirpath="./model/AWA/", save_top_k=1, monitor="val/loss", mode='min')
    trainer = pl.Trainer(max_epochs=MAXEPOCHS, accelerator='auto', devices="auto", callbacks=[checkpoint_cb, EarlyStopping(monitor="val/loss", mode="min", patience=2)])
    trainer.fit(model=model, train_dataloaders=trainDl, val_dataloaders=valDl) 
    model = ConceptMemoryTrees.load_from_checkpoint(checkpoint_cb.best_model_path, backbone=backBone, embeddingSize=EMB_SIZE, treeEmbeddingSize=TREE_EMB_SIZE, treeDecoderNbLayers=TREEDECODERLAYERS, nConcepts=nConcepts, nOutputs=nOutputs, batchSize=batchSize, lr=0.001, treeDepth=depth, memorySize=memorySize, dropout=False, preventFullyFalsePaths=False, distribution_weight=0, hashedSelector=False, treeLearner="indeptendentFPT")
    try:
        model.printTrees()
    except Exception as e:
        print("Error printing trees:", e)
    
    trainer.test(model=model, dataloaders=testDl, verbose=True)


def runExperiment2():
    #HYPER
    EMB_SIZE = 256
    TREE_EMB_SIZE = EMB_SIZE
    TREEDECODERLAYERS = 3
    nConcepts = 85
    nOutputs = 50
    batchSize = 32
    depth = 5
    memorySize = 100 
    lr = 0.001
    MAXEPOCHS = 50
    
    backBone = AWALinearBackbone(input_dim=2048, output_dim=EMB_SIZE)


    trainDl, valDl, testDl = getAWA2EmbeddingDataloaders(embeddings_file="/Users/alecvandeuren/Thesis/src/data/raw/AWA/awa_embeddings.pt", root_dir="/Users/alecvandeuren/Thesis/src/data/raw/AWA", batch_size=batchSize)
    #PHASE 1: train with dropout max
    hists = []
    model = ConceptMemoryTrees(backBone, EMB_SIZE, TREE_EMB_SIZE, TREEDECODERLAYERS, nConcepts, nOutputs, batchSize=batchSize, lr=lr, treeDepth=depth, memorySize=memorySize, dropout=True, dropoutP=1,preventFullyFalsePaths=False, distribution_weight=0, useDescendingLR=True, useALTLOSS=False, HashedSelector=True, treeLearner="indeptendentFPT")
    checkpoint_cb = ModelCheckpoint(dirpath="./model/AWA/", save_top_k=1, monitor="val/loss", mode='min')        
    checkpoint_cb = ModelCheckpoint(dirpath="./model/AWA/", save_top_k=1, monitor="val/loss", mode='min')        
    trainer = pl.Trainer(max_epochs=MAXEPOCHS, accelerator='auto', devices="auto", callbacks=[checkpoint_cb, EarlyStopping(monitor="val/loss", mode="min", patience=5)])
    # if disable validation step, no epoch completed
    trainer.fit(model=model, train_dataloaders=trainDl, val_dataloaders=valDl)       
    combined_tensor = reduce(torch.logical_or, model.maskMap.values())

    print(combined_tensor)
    try:
        model.printTrees()
    except Exception as e:
        print("Error printing trees:", e)
    model = ConceptMemoryTrees.load_from_checkpoint(checkpoint_cb.best_model_path, backbone=backBone, embeddingSize=EMB_SIZE, treeEmbeddingSize=TREE_EMB_SIZE, treeDecoderNbLayers=TREEDECODERLAYERS, nConcepts=nConcepts, nOutputs=nOutputs, batchSize=batchSize, lr=lr/10, treeDepth=depth, memorySize=memorySize, dropout=False, preventFullyFalsePaths=False, distribution_weight=0, useDescendingLR=False, useALTLOSS=False, HashedSelector=False,  treeLearner="indeptendentFPT")
    model.NoSelector = False

    # PHASE 2: train without dropout, but initialize with the weights of the model trained with dropout
    checkpoint_cb = ModelCheckpoint(dirpath="./model/AWA/", save_top_k=1, monitor="val/loss", mode='min')
    trainer = pl.Trainer(max_epochs=MAXEPOCHS, accelerator='auto', devices="auto", callbacks=[checkpoint_cb, EarlyStopping(monitor="val/loss", mode="min", patience=5)])
    trainer.fit(model=model, train_dataloaders=trainDl, val_dataloaders=valDl) 
    model = ConceptMemoryTrees.load_from_checkpoint(checkpoint_cb.best_model_path, backbone=backBone, embeddingSize=EMB_SIZE, treeEmbeddingSize=TREE_EMB_SIZE, treeDecoderNbLayers=TREEDECODERLAYERS, nConcepts=nConcepts, nOutputs=nOutputs, batchSize=batchSize, lr=lr/4, treeDepth=depth, memorySize=memorySize, dropout=False, preventFullyFalsePaths=False, distribution_weight=0, hashedSelector=False, treeLearner="indeptendentFPT")
    try:
        model.printTrees()
    except Exception as e:
        print("Error printing trees:", e)
    
    trainer.test(model=model, dataloaders=testDl, verbose=True)


def runExperiment3():
    #HYPER
    EMB_SIZE = 256
    TREE_EMB_SIZE = EMB_SIZE
    TREEDECODERLAYERS = 3
    nConcepts = 85
    nOutputs = 50
    batchSize = 32
    depth = 5
    memorySize = 50 
    lr = 0.001
    MAXEPOCHS = 50
    
    backBone = AWALinearBackbone(input_dim=2048, output_dim=EMB_SIZE)


    trainDl, valDl, testDl = getAWA2EmbeddingDataloaders(embeddings_file="/Users/alecvandeuren/Thesis/src/data/raw/AWA/awa_embeddings.pt", root_dir="/Users/alecvandeuren/Thesis/src/data/raw/AWA", batch_size=batchSize)
    #PHASE 1: train with dropout max
    hists = []
    model = simpleAbstractModel(backBone, EMB_SIZE, TREE_EMB_SIZE, TREEDECODERLAYERS, nConcepts, nOutputs, batchSize=batchSize, lr=lr, treeDepth=depth, memorySize=memorySize, dropout=True, dropoutP=1,preventFullyFalsePaths=False, distribution_weight=0, useDescendingLR=True, useALTLOSS=False, HashedSelector=True, treeLearner="indeptendentFPT")
    checkpoint_cb = ModelCheckpoint(dirpath="./model/AWA/", save_top_k=1, monitor="val/loss", mode='min')        
    checkpoint_cb = ModelCheckpoint(dirpath="./model/AWA/", save_top_k=1, monitor="val/loss", mode='min')        
    trainer = pl.Trainer(max_epochs=MAXEPOCHS, accelerator='auto', devices="auto", callbacks=[checkpoint_cb, EarlyStopping(monitor="val/loss", mode="min", patience=5)])
    # if disable validation step, no epoch completed
    trainer.fit(model=model, train_dataloaders=trainDl, val_dataloaders=valDl)       
    combined_tensor = reduce(torch.logical_or, model.maskMap.values())

    print(combined_tensor)
    try:
        model.printTrees()
    except Exception as e:
        print("Error printing trees:", e)
    model = simpleAbstractModel.load_from_checkpoint(checkpoint_cb.best_model_path, backbone=backBone, embeddingSize=EMB_SIZE, treeEmbeddingSize=TREE_EMB_SIZE, treeDecoderNbLayers=TREEDECODERLAYERS, nConcepts=nConcepts, nOutputs=nOutputs, batchSize=batchSize, lr=lr/10, treeDepth=depth, memorySize=memorySize, dropout=False, preventFullyFalsePaths=False, distribution_weight=0, useDescendingLR=False, useALTLOSS=False, HashedSelector=False,  treeLearner="indeptendentFPT")
    model.NoSelector = False

    # PHASE 2: train without dropout, but initialize with the weights of the model trained with dropout
    checkpoint_cb = ModelCheckpoint(dirpath="./model/AWA/", save_top_k=1, monitor="val/loss", mode='min')
    trainer = pl.Trainer(max_epochs=MAXEPOCHS, accelerator='auto', devices="auto", callbacks=[checkpoint_cb, EarlyStopping(monitor="val/loss", mode="min", patience=5)])
    trainer.fit(model=model, train_dataloaders=trainDl, val_dataloaders=valDl) 
    model = simpleAbstractModel.load_from_checkpoint(checkpoint_cb.best_model_path, backbone=backBone, embeddingSize=EMB_SIZE, treeEmbeddingSize=TREE_EMB_SIZE, treeDecoderNbLayers=TREEDECODERLAYERS, nConcepts=nConcepts, nOutputs=nOutputs, batchSize=batchSize, lr=lr/10, treeDepth=depth, memorySize=memorySize, dropout=False, preventFullyFalsePaths=False, distribution_weight=0, hashedSelector=False, treeLearner="indeptendentFPT")
    try:
        model.printTrees()
    except Exception as e:
        print("Error printing trees:", e)
    
    trainer.test(model=model, dataloaders=testDl, verbose=True)