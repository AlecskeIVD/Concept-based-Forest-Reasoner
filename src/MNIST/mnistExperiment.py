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



def runExperiment():
    maxDigit = 10
    numberDigits = 2
    embSize = 128
    embTreeSize = 128
    treeDecoderLayers = 3
    nConcepts = numberDigits * maxDigit
    nOutputs = maxDigit * numberDigits - numberDigits + 1
    batchSize = 32
    depth = 3
    memorySize = 50 
    #memorySize = 10

    backbone = MNISTEncoder(emb_size=128, number_digits=2)

    model = ConceptMemoryTrees(backbone, embSize, embTreeSize, treeDecoderLayers, nConcepts, nOutputs, batchSize=batchSize, lr=0.001, treeDepth=depth, memorySize=memorySize, dropout=True, preventFullyFalsePaths=False, dropoutP=0.05, distribution_weight=0)

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
    train_loader = DataLoader(TensorDataset(x_train, c_train, y_train), batch_size=batchSize, shuffle=False, num_workers=7, persistent_workers=True)
    test_loader = DataLoader(TensorDataset(x_test, c_test, y_test), batch_size=batchSize,  num_workers=7, persistent_workers=True)
    val_loader = DataLoader(TensorDataset(x_val, c_val, y_val), batch_size=batchSize,  num_workers=7, persistent_workers=True)


    hists = []
    for w in [0.001, 0.005, 0.01, 0.1, 0.5, 1, 5]:
        model = ConceptMemoryTrees(backbone, embSize, embTreeSize, treeDecoderLayers, nConcepts, nOutputs, batchSize=batchSize, lr=0.001, treeDepth=depth, memorySize=memorySize, dropout=False, preventFullyFalsePaths=False, distribution_weight=w)
        checkpoint_cb = ModelCheckpoint(dirpath="./model/MNIST/", save_top_k=1, monitor="val/loss", mode='min')

        trainer = pl.Trainer(max_epochs=100, accelerator='auto', devices="auto", callbacks=[checkpoint_cb, EarlyStopping(monitor="val/loss", mode="min", patience=100)])
        # if disable validation step, no epoch completed
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # evaluate on test set
        model = ConceptMemoryTrees.load_from_checkpoint(checkpoint_cb.best_model_path, backbone=backbone, embeddingSize=embSize, treeEmbeddingSize=embTreeSize, treeDecoderNbLayers=treeDecoderLayers, nConcepts=nConcepts, nOutputs=nOutputs, batchSize=batchSize, lr=0.0001, treeDepth=depth, memorySize=memorySize, dropout=False, preventFullyFalsePaths=False)
        #results = trainer.test(model=model, dataloaders=test_loader)
        #print("Test results:", results)

        model.printTrees()

        test_loader = DataLoader(TensorDataset(x_test, c_test, y_test), batch_size=1,  num_workers=7, persistent_workers=True)
        #model.runTestLoop(test_loader, numCorrect=5, numFalse=5)
        hists.append(model.collectHistoLeafs(test_loader))
        # hist is a list which contains for each leafIndex how often it was used
        # plot histograms
    for hist in hists:
        print(hist)



def plotMnist():
    import matplotlib.pyplot as plt
    import numpy as np
    from torchvision import datasets, transforms

    # Load MNIST dataset
    mnist = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    # plot 2 images back to back
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    for i in range(2):
        img, label = mnist[i]
        axs[i].imshow(img.squeeze(), cmap='gray')
        axs[i].axis('off')
    plt.tight_layout()
    plt.show()

def runExperimentOnePhase():
    maxDigit = 10
    numberDigits = 2
    embSize = 128
    embTreeSize = 128
    treeDecoderLayers = 3
    nConcepts = numberDigits * maxDigit
    nOutputs = maxDigit * numberDigits - numberDigits + 1
    batchSize = 32
    depth = 3
    memorySize = 50 
    #memorySize = 10

    backbone = MNISTEncoder(emb_size=128, number_digits=2)

    model = ConceptMemoryTrees(backbone, embSize, embTreeSize, treeDecoderLayers, nConcepts, nOutputs, batchSize=batchSize, lr=0.001, treeDepth=depth, memorySize=memorySize, dropout=True, preventFullyFalsePaths=False, dropoutP=0.05, distribution_weight=0, treeLearner="indeptendentFPT")

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
    train_loader = DataLoader(TensorDataset(x_train, c_train, y_train), batch_size=batchSize, shuffle=False, num_workers=7, persistent_workers=True)
    test_loader = DataLoader(TensorDataset(x_test, c_test, y_test), batch_size=batchSize,  num_workers=7, persistent_workers=True)
    val_loader = DataLoader(TensorDataset(x_val, c_val, y_val), batch_size=batchSize,  num_workers=7, persistent_workers=True)

    checkpoint_cb = ModelCheckpoint(dirpath="./model/MNIST/", save_top_k=1, monitor="val/loss", mode='min')        
    trainer = pl.Trainer(max_epochs=100, accelerator='auto', devices="auto", callbacks=[checkpoint_cb, EarlyStopping(monitor="val/loss", mode="min", patience=100)])
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    model = ConceptMemoryTrees.load_from_checkpoint(checkpoint_cb.best_model_path, backbone=backbone, embeddingSize=embSize, treeEmbeddingSize=embTreeSize, treeDecoderNbLayers=treeDecoderLayers, nConcepts=nConcepts, nOutputs=nOutputs, batchSize=batchSize, lr=0.0001, treeDepth=depth, memorySize=memorySize, dropout=False, preventFullyFalsePaths=False, distribution_weight=0, useDescendingLR=True, useALTLOSS=False, HashedSelector=False, treeLearner="indeptendentFPT")
    model.printTrees()
    test_loader = DataLoader(TensorDataset(x_test, c_test, y_test), batch_size=1,  num_workers=7, persistent_workers=True)
    model.runTestLoop(test_loader, numCorrect=5, numFalse=5)





def twoPhaseTraining():
    # first train with dropout, then without dropout
    maxDigit = 10
    numberDigits = 2
    embSize = 128
    embTreeSize = 512
    treeDecoderLayers = 3
    nConcepts = numberDigits * maxDigit
    nOutputs = maxDigit * numberDigits - numberDigits + 1
    batchSize = 32
    depth = 3
    memorySize = 50 
    #memorySize = 10
    MAXEPOCHS = 100

    backbone = MNISTEncoder(emb_size=128, number_digits=2)


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
    train_loader = DataLoader(TensorDataset(x_train, c_train, y_train), batch_size=batchSize, shuffle=False, num_workers=7, persistent_workers=True)
    test_loader = DataLoader(TensorDataset(x_test, c_test, y_test), batch_size=batchSize,  num_workers=7, persistent_workers=True)
    val_loader = DataLoader(TensorDataset(x_val, c_val, y_val), batch_size=batchSize,  num_workers=7, persistent_workers=True)

    #PHASE 1: train with dropout max
    hists = []
    model = ConceptMemoryTrees(backbone, embSize, embTreeSize, treeDecoderLayers, nConcepts, nOutputs, batchSize=batchSize, lr=0.001, treeDepth=depth, memorySize=memorySize, dropout=True, dropoutP=1,preventFullyFalsePaths=False, distribution_weight=0, useDescendingLR=True, useALTLOSS=False, HashedSelector=True, treeLearner="indeptendentFPT")
    checkpoint_cb = ModelCheckpoint(dirpath="./model/MNIST/", save_top_k=1, monitor="val/loss", mode='min')        
    trainer = pl.Trainer(max_epochs=MAXEPOCHS, accelerator='auto', devices="auto", callbacks=[checkpoint_cb, EarlyStopping(monitor="val/loss", mode="min", patience=5)])
    # if disable validation step, no epoch completed
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)       
    combined_tensor = reduce(torch.logical_or, model.maskMap.values())

    print(combined_tensor)
    model.printTrees()
    model = ConceptMemoryTrees.load_from_checkpoint(checkpoint_cb.best_model_path, backbone=backbone, embeddingSize=embSize, treeEmbeddingSize=embTreeSize, treeDecoderNbLayers=treeDecoderLayers, nConcepts=nConcepts, nOutputs=nOutputs, batchSize=batchSize, lr=0.0001, treeDepth=depth, memorySize=memorySize, dropout=False, preventFullyFalsePaths=False, distribution_weight=0, useDescendingLR=True, useALTLOSS=False, HashedSelector=False,  treeLearner="indeptendentFPT")
    model.NoSelector = False

    # PHASE 2: train without dropout, but initialize with the weights of the model trained with dropout
    checkpoint_cb = ModelCheckpoint(dirpath="./model/MNIST/", save_top_k=1, monitor="val/loss", mode='min')
    trainer = pl.Trainer(max_epochs=MAXEPOCHS, accelerator='auto', devices="auto", callbacks=[checkpoint_cb, EarlyStopping(monitor="val/loss", mode="min", patience=5)])
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader) 
    model = ConceptMemoryTrees.load_from_checkpoint(checkpoint_cb.best_model_path, backbone=backbone, embeddingSize=embSize, treeEmbeddingSize=embTreeSize, treeDecoderNbLayers=treeDecoderLayers, nConcepts=nConcepts, nOutputs=nOutputs, batchSize=batchSize, lr=0.001, treeDepth=depth, memorySize=memorySize, dropout=False, preventFullyFalsePaths=False, distribution_weight=0, hashedSelector=False, treeLearner="indeptendentFPT")

    model.printTrees()
    test_loader = DataLoader(TensorDataset(x_test, c_test, y_test), batch_size=1,  num_workers=7, persistent_workers=True)
    model.runTestLoop(test_loader, numCorrect=5, numFalse=5)


    #results = trainer.test(model=model, dataloaders=test_loader)
    #print("Test results:", results)        model.printTrees()        
    # test_loader = DataLoader(TensorDataset(x_test, c_test, y_test), batch_size=1,  num_workers=7, persistent_workers=True)
    #model.runTestLoop(test_loader, numCorrect=5, numFalse=5)
    hists.append(model.collectHistoLeafs(test_loader))
    # hist is a list which contains for each leafIndex how often it was used
    # plot histograms
    for hist in hists:
        print(hist)


if __name__ == '__main__':
    print("dropout experiment")
    #runExperiment()
    # RUN EXAMPLES TO DEBUG
    # Check effect van aantal bomen op accuracy
    # cub
    # leer betere regels
    plotMnist()

    # Reinitiliaze selector
    # Check if learning rate gets reinitiialzed

    # recreate dataset with produced embeddings from pretrained frozen model to speed up training
    # sanity check: simple MLP

    # If everything fails, use XGBOOST to learn the tree structure and throw gradient based learning out of the window. We can still use the concept predictor and selector, but we will not be able to do end to end training anymore. This is a last resort, but it might be necessary if we cannot get the gradient based learning to work.