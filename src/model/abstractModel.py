from abc import abstractmethod
import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, matthews_corrcoef
import pytorch_lightning as pl
from torch.nn.functional import binary_cross_entropy, cross_entropy
import torch
from collections import Counter
from .treeUtils import *
from sklearn.metrics import r2_score


"""
This class will represent the abstract Concept Memory Trees
"""
class ConceptMemoryTrees(pl.LightningModule):
    _SUPPORTEDTREES = {"fastProbabilisticTree", "indeptendentFPT", "regressionFPT", "predeterminedTrees"}
    def __init__(self, backbone, embeddingSize, treeEmbeddingSize, treeDecoderNbLayers, nConcepts, nOutputs, batchSize=32, classification=True, temperature: float = 1., treeDepth=3, memorySize=10, treeLearner="fastProbabilisticTree", concept_weight=1.0, lr=1e-3, dropout=False, preventFullyFalsePaths: bool = False, distribution_weight = 0.01, dropoutP=0, useDescendingLR=True, useALTLOSS = False, HashedSelector = False, predeterminedTrees = False, givenTrees = None):
        """
        Function to initiliaze the model.
        backbone: pytorch model which will produce embeddings from the input data
        embeddingSize: size of the embedding produced by the backbone
        treeDecoderNbLayers: number of layers in the rule decoder network
        treeEmbeddingSize: size of the embedding used for the decision trees
        nConcepts: number of concepts to be calculated
        nOutputs: number of output classes to be predicted
        classification: bool which represents if we are doing classification or regression
        temperature: temperature for softmaxes
        treeDepth: the height of the decision trees
        memorySize: number of trees that will be in our memory
        treeLearner: Method to learn the decision trees with gradient descent. Current options are "fastProbabilisticTree", "indeptendentFPT", "regressionFPT", "predeterminedTrees"
        lr: learning rate for the optimizer
        concept_weight: weight for the concept loss in the total loss
        """
        # Check inputs
        assert embeddingSize > 0
        assert nConcepts > 0
        assert nOutputs > 0
        assert treeDepth > 0
        assert memorySize > 0
        assert treeLearner in self._SUPPORTEDTREES
        assert lr > 0

        super().__init__()
        self.backbone = backbone
        self.embeddingSize = embeddingSize
        self.treeEmbeddingSize = treeEmbeddingSize
        self.nConcepts = nConcepts
        self.nClasses = nOutputs
        self.classification = classification
        self.temperature = temperature
        self.treeDepth = treeDepth
        self.memorySize = memorySize
        self.treeLearner = treeLearner
        self.concept_weight = concept_weight
        self.lr = lr
        self.dropout = dropout
        self.preventFullyFalsePaths = preventFullyFalsePaths
        self.distributionWeight = distribution_weight
        self.KLDivergence = torch.nn.KLDivLoss(log_target=True)
        self.dropoutP = dropoutP
        self.useDescendingLR = useDescendingLR
        self.useALTLOSS = useALTLOSS
        self.altLoss = MixtureLoss()
        self.NoSelector = False
        self.hashedSelector = HashedSelector
        self.maskMap = {}
        self.sizeOfMap = memorySize * 20 # tweak
        self.numTreesPerEntry = memorySize // 10

        self.predeterminedTrees = predeterminedTrees
        self.givenTrees = givenTrees

        if self.treeLearner == "predeterminedTrees" and not self.predeterminedTrees:
            raise ValueError("If treeLearner is predeterminedTrees, predeterminedTrees must be True")
        if self.treeLearner != "predeterminedTrees" and self.predeterminedTrees:
            raise ValueError("If treeLearner is not predeterminedTrees, predeterminedTrees must be False")
        if self.predeterminedTrees and self.givenTrees is None:
            raise ValueError("If predeterminedTrees is True, givenTrees must be provided")


        if treeLearner == "GradTree":
            self.oldVersion = True
            self.treeVarsSize = nConcepts * (2**treeDepth - 1) + (2**treeDepth) * nOutputs # replace with tensor 
            self.numInternalNodesVariables = (2**treeDepth - 1) * nConcepts # RESHAPE OUTPUT EMBEDDINGTRANSLATOR TO DIFFERENT TENSORS
            self.dtEmbeddings = torch.nn.Embedding(self.memorySize, self.treeEmbeddingSize)
            self.decisionTree = GradTree()
            self.embeddingTranslator = TreeDecoder(treeEmbeddingSize, nConcepts, nOutputs, treeDepth, treeDecoderNbLayers, self.oldVersion)
        elif treeLearner == "ProbabilisticTree":
            self.oldVersion = True
            self.treeVarsSize = nConcepts * (2**treeDepth - 1) + (2**treeDepth) * nOutputs # replace with tensor 
            self.numInternalNodesVariables = (2**treeDepth - 1) * nConcepts # RESHAPE OUTPUT EMBEDDINGTRANSLATOR TO DIFFERENT TENSORS
            self.dtEmbeddings = torch.nn.Embedding(self.memorySize, self.treeEmbeddingSize)
            self.decisionTree = ProbabilisticTree()
            self.embeddingTranslator = TreeDecoder(treeEmbeddingSize, nConcepts, nOutputs, treeDepth, treeDecoderNbLayers, self.oldVersion)
        elif treeLearner == "fastProbabilisticTree":
            self.oldVersion = False
            self.treeVarsSize = nConcepts * (2**treeDepth - 1) + (2**treeDepth) * nOutputs # replace with tensor 
            self.numInternalNodesVariables = (2**treeDepth - 1) * nConcepts # RESHAPE OUTPUT EMBEDDINGTRANSLATOR TO DIFFERENT TENSORS
            self.dtEmbeddings = torch.nn.Embedding(self.memorySize, self.treeEmbeddingSize)
            self.decisionTree = fastProbabilisticTree(treeDepth, batchSize, memorySize) # memory size trees, batch size 1 for evaluation
            self.embeddingTranslator = TreeDecoder(treeEmbeddingSize, nConcepts, nOutputs, treeDepth, treeDecoderNbLayers, self.oldVersion)
        elif treeLearner == "indeptendentFPT":
            self.oldVersion = False
            self.numInternalNodesVariables = (2**treeDepth - 1) * nConcepts
            self.nLeafVars = 2**treeDepth * nOutputs
            self.dtEmbeddings = torch.nn.Embedding(self.memorySize, self.numInternalNodesVariables + self.nLeafVars)
            self.decisionTree = fastProbabilisticTree(treeDepth, batchSize, memorySize)
            self.embeddingTranslator = IndependentTreeDecoder(treeDepth, self.numInternalNodesVariables, self.nLeafVars, nConcepts, nOutputs)
        elif treeLearner == "regressionFPT":
            self.oldVersion = False
            self.numInternalNodesVariables = (2**treeDepth - 1) * nConcepts
            self.nLeafVars = 2**treeDepth
            self.dtEmbeddings = torch.nn.Embedding(self.memorySize, self.numInternalNodesVariables + self.nLeafVars)
            self.decisionTree = regressionFPT(treeDepth, batchSize, memorySize)
            self.embeddingTranslator = RegressionTreeDecoder(treeDepth, self.numInternalNodesVariables, self.nLeafVars, nConcepts)

        elif treeLearner == "predeterminedTrees":
            self.oldVersion = False
            self.numInternalNodesVariables = (2**treeDepth - 1) * nConcepts
            self.nLeafVars = 2**treeDepth * nOutputs
            self.dtEmbeddings = torch.nn.Embedding(self.memorySize, self.numInternalNodesVariables + self.nLeafVars) # aren't used
            self.splitVars, self.leafValues = givenTrees
            self.embeddingTranslator = PredeterminedTreeDecoder(self.splitVars, self.leafValues)
            self.decisionTree = fastProbabilisticTree(treeDepth, batchSize, memorySize)
            

        else:
            # Should be impossible
            self.oldVersion = True
            self.treeVarsSize = 0
            self.numInternalNodesVariables = 0
            self.dtEmbeddings = torch.nn.Embedding(self.treeEmbeddingSize, self.memorySize) # Every column in this matrix is embedding of one of the decision trees
            self.decisionTree = DecisionTree()
            self.embeddingTranslator = torch.nn.Sequential()

        

        self.treeSelector = torch.nn.Sequential(
            torch.nn.Linear(embeddingSize, embeddingSize),
            torch.nn.ReLU(),
            torch.nn.Linear(embeddingSize, memorySize), # size batchSize, memorySize
        )

        self.conceptPredictor = torch.nn.Sequential(
            torch.nn.Linear(self.embeddingSize, self.embeddingSize),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embeddingSize, self.embeddingSize),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embeddingSize, self.embeddingSize),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embeddingSize, nConcepts),# size batchSize, nConcepts
        )

        

    def forward(self, x, train: bool = True):
        batch_x, batch_c, batch_y = x
        batch_size = batch_x.size(0)

        emb = self.backbone(batch_x)  # batch, backbone_out_shape
        cPredProb = self.conceptPredictor(emb).sigmoid()  # batch, nb_concepts

        # Perform droput by rarely zeroing out embedding. Hopefully promotes more accurate trees
        if not self.hashedSelector:
            if self.NoSelector:
                treeSelProbsLogits_bt = torch.zeros((batch_size, self.memorySize), device=emb.device)
                treeSelProbs_bt = torch.ones((batch_size, self.memorySize), device=emb.device) / self.memorySize
            elif self.dropout and random.random() < self.dropoutP:
                input = torch.zeros(emb.size()).to(emb.device)
                treeSelProbsLogits_bt = self.treeSelector(input).view(batch_size, self.memorySize)
                # Dropout half trees by setting their logits to a very low value, so that after softmax they have close to zero probability
                dropout_mask = torch.rand(treeSelProbsLogits_bt.size()).to(emb.device) < 0.5
                treeSelProbsLogits_bt = treeSelProbsLogits_bt.masked_fill(dropout_mask, -1e9)
                treeSelProbs_bt = treeSelProbsLogits_bt.softmax(dim=-1)
            else:
                treeSelProbsLogits_bt = self.treeSelector(emb).view(batch_size, self.memorySize)
                treeSelProbs_bt = treeSelProbsLogits_bt.softmax(dim=-1)  # batch, nb_trees
        else:
            # for each concept vector
            masks = []
            for i in range(batch_size):
                # hash concept vector
                concept_vector = cPredProb[i].detach().cpu().numpy() > 0.5
                hashedVal = hash(tuple(concept_vector)) % self.sizeOfMap
                if hashedVal not in self.maskMap:
                    # create random mask with numTreesPerEntry ones and the rest zeros
                    mask = torch.zeros(self.memorySize, dtype=torch.bool)
                    mask[:self.numTreesPerEntry] = 1
                    mask = mask[torch.randperm(self.memorySize)]
                    self.maskMap[hashedVal] = mask.to(emb.device)
                else:
                    mask = self.maskMap[hashedVal]
                masks.append(mask.to(emb.device))
            masks = torch.stack(masks, dim=0) # batch, memorySize
            input = torch.zeros(emb.size()).to(emb.device)

            #treeSelProbsLogits_bt = self.treeSelector(input).view(batch_size, self.memorySize)
            treeSelProbsLogits_bt = torch.zeros((batch_size, self.memorySize), device=emb.device)
            
            treeSelProbsLogits_bt = treeSelProbsLogits_bt.masked_fill(~masks, -1e9)
            treeSelProbs_bt = treeSelProbsLogits_bt.softmax(dim=-1)
                

                

        device = emb.device  # same device as backbone output

        treeVarsProbs_tcv, leafValues_tcv = self.embeddingTranslator(self.dtEmbeddings.weight)

        if self.preventFullyFalsePaths:
            M, L, O = leafValues_tcv.shape  # memorySize, nLeaves, nOutputs

            mask = torch.zeros_like(leafValues_tcv)
            mask[:,0,:] = torch.ones(O)
            mask = mask.bool().to(device)
            # shape: (nLeaves, 1), broadcasts over memorySize and nOutputs
            # Uniform value
            uniform_val = 1.0 / self.nClasses

            # Apply in-place torch.where using broadcasting
            leafValues_tcv = torch.where(mask, uniform_val, leafValues_tcv)

        #treeVarsProbs_btv = treeVarsProbs_tcv.unsqueeze(0).expand(batch_size, -1, -1)  # batch, nb_trees, nb_variablesTree
        # aanpassen naar verschillende tensors voor internal nodes en leafs

        # Evaluate every tree in the memory
        if self.oldVersion:
            outputs = torch.zeros((batch_size, self.nClasses)).to(device)  # Initialize output tensor
            for tree in range(self.memorySize):
                # Extract variables for this tree
                treeVars = treeVarsProbs_tcv[tree, :]
                splitVars = treeVars.to(device)
                leafValues = leafValues_tcv[tree,:].view(2**self.treeDepth, self.nClasses).to(device)

                # Evaluate tree for all entries in batch
                treeOutput, leafProbs_bml = self.decisionTree.evaluate(cPredProb, splitVars, leafValues, self.nClasses, self.treeDepth)  # batch, nClasses

                # Weight output by probability of selecting this tree
                outputs = outputs + treeSelProbs_bt[:, tree].unsqueeze(1) * treeOutput  # batch, nClasses
        else:
            if self.treeLearner == "regressionFPT":
                outputs_bm, leafProbs_bml = self.decisionTree.evaluate(cPredProb, treeVarsProbs_tcv, leafValues_tcv, self.treeDepth) # (B, M)
                outputs = torch.einsum('bm, bm->b', treeSelProbs_bt, outputs_bm) # (B)
                outputs_bmo = None
            else:
                outputs_bmo, leafProbs_bml = self.decisionTree.evaluate(cPredProb, treeVarsProbs_tcv, leafValues_tcv, self.nClasses, self.treeDepth) # (B, M, nOutputs)
                # Weight output by probability of selecting this tree, given by treeSelProbs_bt (B, M)
                outputs = torch.einsum('bm, bmo->bo', treeSelProbs_bt, outputs_bmo) # (B, nOutputs)
                outputs_bm = None

        return {"y_pred_probs_bo": outputs, "c_pred_probs_bc": cPredProb, "leafProbs_bml": leafProbs_bml, "treeSelProbs_bt":treeSelProbs_bt, "outputs_bmo": outputs_bmo, "treeSelProbsLogits_bt": treeSelProbsLogits_bt, "outputs_bm": outputs_bm}
    

    def training_step(self, batch, batch_idx):
        batch_x, batch_c, batch_y = batch

        out = self.forward(batch)
        y_pred_probs_bo = out["y_pred_probs_bo"]
        c_pred_probs_bc = out["c_pred_probs_bc"]
        leafProbs_bml = out["leafProbs_bml"]
        treeSelProbs_bm1 = out["treeSelProbs_bt"].unsqueeze(-1)

        #print("y_pred_probs_bo:", y_pred_probs_bo)
        #print("c_pred_probs_bc:", c_pred_probs_bc)
        #print("batch_y:", batch_y)
        #print("batch_c:", batch_c)

        #loss_y = binary_cross_entropy(y_pred_probs_bo, batch_y)
        if self.treeLearner == "regressionFPT":
            loss_y = torch.nn.functional.mse_loss(y_pred_probs_bo, batch_y)
        else:
            if not self.useALTLOSS:
                loss_y = torch.nn.functional.nll_loss(torch.log(y_pred_probs_bo + 1e-8), batch_y.argmax(dim=-1))
            else:
                loss_y = self.altLoss(batch_y, out["outputs_bmo"], torch.nn.functional.log_softmax(out["treeSelProbsLogits_bt"], dim=-1))
        loss_c = binary_cross_entropy(c_pred_probs_bc, batch_c)

        # extra loss als KL-Divergence tussen batch distributie en uniforme verdeling (weighted probs)

        weightedLeafDistribution_bl = torch.sum(leafProbs_bml * treeSelProbs_bm1, dim=1)

        leafLogDistribution=torch.log(torch.mean(weightedLeafDistribution_bl, dim=0)+1e-8)
        targetLogDistribution = torch.log(1/max(leafLogDistribution.size()) * torch.ones_like(leafLogDistribution))

        
        if self.distributionWeight > 0:
            loss_l = self.KLDivergence(leafLogDistribution, targetLogDistribution)

        total = loss_y + self.concept_weight * loss_c #+ self.distributionWeight * loss_l

        self.log("train/loss", total, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/loss_y", loss_y, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("train/loss_c", loss_c, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        #self.log("train/loss_l", loss_l, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        c_pred = (c_pred_probs_bc > 0.5).float()
        #y_pred = (y_pred_probs_bo > 0.5).float()
        y_pred = torch.nn.functional.one_hot(y_pred_probs_bo.argmax(dim=-1), self.nClasses).float()

        with torch.no_grad():
            concept_acc = (c_pred == batch_c).float().mean()
            self.log("train/concept_flat_acc", concept_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        if self.treeLearner != "regressionFPT":

            with torch.no_grad():
                task_acc = (y_pred == batch_y).float().mean()
                self.log("train/task_flat_acc", task_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)

            with torch.no_grad():
                subset_acc = ((y_pred == batch_y).float().prod(dim=-1)).mean()
                self.log("train/task_subset_acc", subset_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)

            # Most important!!!
            with torch.no_grad():
                y_true_mc = batch_y.argmax(dim=-1)
                y_pred_mc = y_pred.argmax(dim=-1)
                multiclass_acc = (y_true_mc == y_pred_mc).float().mean()
                self.log("train/task_multiclass_acc", multiclass_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        else:
            with torch.no_grad():
                # Compute R2 score
                y_true = batch_y.cpu().numpy()
                y_pred_np = y_pred_probs_bo.cpu().numpy()
                r2 = r2_score(y_true, y_pred_np)
                self.log("train/r2_score", r2, on_step=False, on_epoch=True, prog_bar=False, logger=True)


        return total


    def validation_step(self, batch, batch_idx):
        batch_x, batch_c, batch_y = batch

        out = self.forward(batch, train=False)
        y_pred_probs_bo = out["y_pred_probs_bo"]
        c_pred_probs_bc = out["c_pred_probs_bc"]

        #loss_y = binary_cross_entropy(y_pred_probs_bo, batch_y)
        if self.treeLearner == "regressionFPT":
            loss_y = torch.nn.functional.mse_loss(y_pred_probs_bo, batch_y)
        else:
            if not self.useALTLOSS:
                loss_y = torch.nn.functional.nll_loss(torch.log(y_pred_probs_bo + 1e-8), batch_y.argmax(dim=-1))
            else:
                loss_y = self.altLoss(batch_y, out["outputs_bmo"], torch.nn.functional.log_softmax(out["treeSelProbsLogits_bt"], dim=-1))
        loss_c = binary_cross_entropy(c_pred_probs_bc, batch_c)
        total = loss_y + self.concept_weight * loss_c

        self.log("val/loss", total, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/loss_y", loss_y, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val/loss_c", loss_c, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        c_pred = (c_pred_probs_bc > 0.5).float()
        #y_pred = (y_pred_probs_bo > 0.5).float()

        # change y_pred to be all zeros except for 1 at argmax
        y_pred = torch.nn.functional.one_hot(y_pred_probs_bo.argmax(dim=-1), self.nClasses).float()

        with torch.no_grad():
            concept_acc = (c_pred == batch_c).float().mean()
            self.log("val/concept_flat_acc", concept_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        if self.treeLearner != "regressionFPT":
            with torch.no_grad():
                task_acc = (y_pred == batch_y).float().mean()
                self.log("val/task_flat_acc", task_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)

            with torch.no_grad():
                y_true_mc = batch_y.argmax(dim=-1)
                y_pred_mc = y_pred_probs_bo.argmax(dim=-1)
                multiclass_acc = (y_true_mc == y_pred_mc).float().mean()
                self.log("val/task_multiclass_acc", multiclass_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        else:
            with torch.no_grad():
                # Compute R2 score
                y_true = batch_y.cpu().numpy()
                y_pred_np = y_pred_probs_bo.cpu().numpy()
                r2 = r2_score(y_true, y_pred_np)
                self.log("val/r2_score", r2, on_step=False, on_epoch=True, prog_bar=False, logger=True)
                mse = torch.nn.functional.mse_loss(y_pred_probs_bo, batch_y)
                self.log("val/mse", mse, on_step=False, on_epoch=True, prog_bar=False, logger=True)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        if self.useDescendingLR:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.3)
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",   # or "step" for per-batch updates
                "frequency": 1,
                "monitor": "val/loss"  # optional if you want to log or checkpoint based on val loss
            },
        }

    def printTrees(self):
        treeVarsProbs_tcv, leafValues_tcv = self.embeddingTranslator(self.dtEmbeddings.weight) # nb_trees, nb_variablesTree
        out = ""
        for tree in range(self.memorySize):
            print(f"Tree {tree}:")
            treeVars = treeVarsProbs_tcv[tree,: , :]
            splitVars = treeVars.to(self.dtEmbeddings.weight.device)
            leafValues = leafValues_tcv[tree,:].view(2**self.treeDepth, self.nClasses).to(self.dtEmbeddings.weight.device)
            strVal = self.decisionTree.printTree(splitVars, leafValues, self.nConcepts, self.treeDepth)
            out += f"Tree {tree}:\n{strVal}"
            print("---------")
            out += "---------\n"
        return out
    

    def runTestLoop(self, test_loader, numCorrect=5, numFalse=5):
        """"
        Assumes test_loader has batch size of 1
        """
        with torch.no_grad():
            curNumCorrect = 0
            curNumFalse = 0
            dl = iter(test_loader)
            print('\x1b[6;30;42m' + "CORRECT EXAMPLES" + '\x1b[0m')
            while curNumCorrect < numCorrect:
                digits = next(dl)
                digits = (digits[0].to(self.dtEmbeddings.weight.device), digits[1].to(self.dtEmbeddings.weight.device), digits[2].to(self.dtEmbeddings.weight.device))
                out = self.forward(digits) # b, noutputs and b, nConcepts
                if out.get("y_pred_probs_bo").argmax(dim=-1).item() == digits[2].argmax(dim=-1).item():
                    curNumCorrect += 1
                    print("--- Example", curNumCorrect, "---")
                    print("Predicted concepts:", (out.get("c_pred_probs_bc") > 0.5).float())
                    print("True concepts:     ", digits[1])
                    print("Chosen tree:")
                    treeSelProbs_bt = self.treeSelector(self.backbone(digits[0])).view(1, self.memorySize).softmax(dim=-1)
                    index = torch.argmax(treeSelProbs_bt, dim=-1).item()

                    treeVarsProbs_tcv, leafValues_tcv = self.embeddingTranslator(self.dtEmbeddings.weight) # nb_trees, nb_variablesTree
        
                    treeVars = treeVarsProbs_tcv[index,: , :]
                    splitVars = treeVars.to(self.dtEmbeddings.weight.device)
                    leafValues = leafValues_tcv[index,:].view(2**self.treeDepth, self.nClasses).to(self.dtEmbeddings.weight.device)
                    self.decisionTree.printTree(splitVars, leafValues, self.nConcepts, self.treeDepth)
                    print("---------")
            print('\x1b[6;30;41m' + "FALSE EXAMPLES" + '\x1b[0m')
            while curNumFalse < numFalse:
                digits = next(dl)
                digits = (digits[0].to(self.dtEmbeddings.weight.device), digits[1].to(self.dtEmbeddings.weight.device), digits[2].to(self.dtEmbeddings.weight.device))
                out = self.forward(digits) # b, noutputs and b, nConcepts
                if out.get("y_pred_probs_bo").argmax(dim=-1).item() != digits[2].argmax(dim=-1).item():
                    curNumFalse += 1
                    print("--- Example", curNumFalse, "---")
                    print("Predicted concepts:", (out.get("c_pred_probs_bc") > 0.5).float())
                    print("True concepts:     ", digits[1])
                    print("Chosen tree:")
                    treeSelProbs_bt = self.treeSelector(self.backbone(digits[0])).view(1, self.memorySize).softmax(dim=-1)
                    index = torch.argmax(treeSelProbs_bt, dim=-1).item()

                    treeVarsProbs_tcv, leafValues_tcv = self.embeddingTranslator(self.dtEmbeddings.weight) # nb_trees, nb_variablesTree
        
                    treeVars = treeVarsProbs_tcv[index,: , :]
                    splitVars = treeVars.to(self.dtEmbeddings.weight.device)
                    leafValues = leafValues_tcv[index,:].view(2**self.treeDepth, self.nClasses).to(self.dtEmbeddings.weight.device)
                    self.decisionTree.printTree(splitVars, leafValues, self.nConcepts, self.treeDepth)
                    print("---------")
    
    def collectHistoLeafs(self, test_loader):
        with torch.no_grad():
            leafValueCounters = [0 for _ in range(2**self.treeDepth)]
            dl = iter(test_loader)
            for digits in dl:
                digits = (digits[0].to(self.dtEmbeddings.weight.device), digits[1].to(self.dtEmbeddings.weight.device), digits[2].to(self.dtEmbeddings.weight.device))
                emb = self.backbone(digits[0])
                treeSelProbs_bt = self.treeSelector(emb).view(1, self.memorySize).softmax(dim=-1)
                index = torch.argmax(treeSelProbs_bt, dim=-1).item()
                treeVarsProbs_tcv, leafValues_tcv = self.embeddingTranslator(self.dtEmbeddings.weight) # nb_trees, nb_variablesTree
                # calculate which leaf is most likely in most likely tree
                treeVars = treeVarsProbs_tcv[index,: , :]
                splitVars = treeVars.to(self.dtEmbeddings.weight.device)
                # calc leaf probs:
                leafProbs_bml = self.decisionTree.collectLeafProbs(digits[1], splitVars.unsqueeze(0), self.treeDepth) # (B, M, L)
                leafProbs_ml = leafProbs_bml[0,:,:] # (M,,L)
                mostLikelyLeaf = torch.argmax(leafProbs_ml, dim=-1) #
                for leaf in mostLikelyLeaf.cpu().numpy().tolist():
                    leafValueCounters[leaf] += 1
        return leafValueCounters

    
    def collectHistoTrees(self, test_loader):
        pass

    def test_step(self, batch, batch_idx):
        batch_x, batch_c, batch_y = batch

        out = self.forward(batch, train=False)
        y_pred_probs_bo = out["y_pred_probs_bo"]
        c_pred_probs_bc = out["c_pred_probs_bc"]


        c_pred = (c_pred_probs_bc > 0.5).float()
        #y_pred = (y_pred_probs_bo > 0.5).float()

        # change y_pred to be all zeros except for 1 at argmax
        y_pred = torch.nn.functional.one_hot(y_pred_probs_bo.argmax(dim=-1), self.nClasses).float()

        with torch.no_grad():
            concept_acc = (c_pred == batch_c).float().mean()
            self.log("test/concept_flat_acc", concept_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        if self.treeLearner != "regressionFPT": 
            with torch.no_grad():
                y_true_mc = batch_y.argmax(dim=-1)
                y_pred_mc = y_pred_probs_bo.argmax(dim=-1)
                multiclass_acc = (y_true_mc == y_pred_mc).float().mean()
                self.log("test/task_multiclass_acc", multiclass_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        else:
            with torch.no_grad():
                # Compute R2 score
                y_true = batch_y.cpu().numpy()
                y_pred_np = y_pred_probs_bo.cpu().numpy()
                r2 = r2_score(y_true, y_pred_np)
                self.log("test/r2_score", r2, on_step=False, on_epoch=True, prog_bar=False, logger=True)
                # MSE
                mse = torch.nn.functional.mse_loss(y_pred_probs_bo, batch_y)
                self.log("test/mse", mse, on_step=False, on_epoch=True, prog_bar=False, logger=True)

                


class TreeDecoder(pl.LightningModule):
    def __init__(self, treeEmbeddingSize, nConcepts, nOutput, depth,treeDecoderNbLayers, reshapeOutputToVectors: bool = True):
        super().__init__()
        self.depth = depth
        self.nConcepts = nConcepts
        self.nOutput = nOutput
        self.reshapeOutputToVectors = reshapeOutputToVectors
        # Sequential for repeated layers
        self.seq = torch.nn.Sequential(
            *[layer for _ in range(treeDecoderNbLayers - 1) 
              for layer in (torch.nn.Linear(treeEmbeddingSize, treeEmbeddingSize), torch.nn.ReLU())]
        )
        # Two final linear layers for separate outputs
        self.out1 = torch.nn.Linear(treeEmbeddingSize, (2**depth-1)*nConcepts) # First one must create tensor of dimension (2**depth - 1), nConcepts
        self.out2 = torch.nn.Linear(treeEmbeddingSize, 2**depth * nOutput) # Second one must create tensor of dimension (2**depth), nOutput

    def forward(self, x: torch.Tensor):
        x = self.seq(x)
        treeVars = self.out1(x).view(-1, (2**self.depth - 1), self.nConcepts).softmax(dim=-1) # 
         # treevars is tensor of shape (memorySize, (2**depth - 1), nConcepts)
         # Now we want to make treevars tensor of shape (memorySize, (2**depth - 1)* nConcepts) by concatenating all the softmaxed vectors
        if self.reshapeOutputToVectors:
            treeVars = treeVars.view(-1, (2**self.depth - 1)*self.nConcepts)
        
        leafValues = self.out2(x).view(-1, 2**self.depth, self.nOutput).softmax(dim=-1)
        if self.reshapeOutputToVectors:
            leafValues = leafValues.view(-1, 2**self.depth * self.nOutput)
        
        return treeVars, leafValues # returns two tensors


class IndependentTreeDecoder(pl.LightningModule):
    def __init__(self, depth, numInternalNodesVariables, numLeafVariables, nConcepts, nOutput):
        super().__init__()
        self.depth = depth
        self.numInternalNodesVariables = numInternalNodesVariables
        self.numLeafVariables = numLeafVariables
        self.nConcepts = nConcepts
        self.nOutput = nOutput

    def forward(self, x: torch.Tensor):
        # x is of shape (memorySize, numInternalNodesVariables + numLeafVariables)
        # split x into tensor 1 of shape (memorySize, numInternalNodesVariables) and tensor 2 of shape (memorySize, numLeafVariables)
        xInternal = x[:, :self.numInternalNodesVariables]
        xLeaf = x[:, self.numInternalNodesVariables:]
        treeVars = xInternal.view(-1, (2**self.depth - 1), self.nConcepts).softmax(dim=-1)
        leafValues = xLeaf.view(-1, 2**self.depth, self.nOutput).softmax(dim=-1)
        return treeVars, leafValues


class RegressionTreeDecoder(pl.LightningModule):
    def __init__(self, depth, numInternalNodesVariables, numLeafVariables, nConcepts):
        super().__init__()
        self.depth = depth
        self.numInternalNodesVariables = numInternalNodesVariables
        self.numLeafVariables = numLeafVariables
        self.nConcepts = nConcepts

    def forward(self, x: torch.Tensor):
        # x is of shape (memorySize, numInternalNodesVariables + numLeafVariables)
        # split x into tensor 1 of shape (memorySize, numInternalNodesVariables) and tensor 2 of shape (memorySize, numLeafVariables)
        xInternal = x[:, :self.numInternalNodesVariables]
        xLeaf = x[:, self.numInternalNodesVariables:]
        treeVars = xInternal.view(-1, (2**self.depth - 1), self.nConcepts).softmax(dim=-1)
        leafValues = xLeaf.view(-1, 2**self.depth)
        return treeVars, leafValues

class PredeterminedTreeDecoder(pl.LightningModule):
    def __init__(self, splitVars, leafValues):
        super().__init__()
        self.register_buffer("splitVars", splitVars)
        self.register_buffer("leafValues", leafValues)

    def forward(self, x: torch.Tensor):
        return self.splitVars, self.leafValues
"""
pytorch lightning optimiseert alle leerbare parameters
piecewise maal + .sum() voor gewogen som
forwards accepts batches
.detach voor trainen om gradient tegen te houden
discreet stopt gradient
over pytorch 
Hardcut experimenteren
Poging 1: bij training ook hardcut van concepts, andere dingen proberen als het niet werkt
"""


"""
Om te beginnen
reduceer digit dataset naar minder digits
https://github.com/daviddebot/CMR/blob/main/experiments/mnist/mnist_dataset.py
gebruik digit limit
leaf outputs met softmax voor classificatie"""

if __name__ == "__main__":
    model = ConceptMemoryTrees(None, 10, 5, 2, 3, 2, treeLearner="fastProbabilisticTree")
    

class MixtureLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y: torch.Tensor, f_x, log_w_x):
        # y is shape (B, nOutputs), f_x is shape (B, M, nOutputs), log_w_x is shape (B, M)
        # reshape f_x to (B*M, nOutputs) and y_repeated to (B*M, nOutputs) for loss calculation
        f_x = f_x.reshape(-1, f_x.shape[-1])
        y_repeated = y.unsqueeze(1).expand(-1, f_x.shape[0] // y.shape[0], -1).reshape(-1, y.shape[-1])
        log_diff = torch.nn.functional.nll_loss(torch.log(f_x + 1e-8), y_repeated.argmax(dim=-1))
        exponent = log_w_x - 0.5 * log_diff
        loss_per_sample = -torch.logsumexp(exponent, dim=1)
        return loss_per_sample.mean()