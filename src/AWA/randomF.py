from collections import deque

import sklearn
from src.AWA.awaDataset import get_awa2_dataloader, get_awa2_dataloaders, getAWA2EmbeddingDataloaders
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, matthews_corrcoef
import pytorch_lightning as pl
from torch.nn.functional import binary_cross_entropy, cross_entropy


def createForest():
    trainDl, valDL, testDL = getAWA2EmbeddingDataloaders("/Users/alecvandeuren/Thesis/src/data/raw/AWA/awa_embeddings.pt","/Users/alecvandeuren/Thesis/src/data/raw/AWA" ,batch_size=1, shuffle=False, num_workers=0)
    depth = [5, 8, 10]
    size = [100, 300, 500]
    featuresTest = []
    labelsTest = []
    featuresTrain = []
    labelsTrain = []
    while len(featuresTrain) < len(trainDl.dataset):
        image, feature, label = trainDl.dataset[len(featuresTrain)]
        featuresTrain.append(feature.numpy())
        labelsTrain.append(label)
    featuresTrain = np.array(featuresTrain)
    labelsTrain = np.array(labelsTrain)
    featuresVal = []
    labelsVal = []
    while len(featuresVal) < len(valDL.dataset):
        image, feature, label = valDL.dataset[len(featuresVal)]
        featuresVal.append(feature.numpy())
        labelsVal.append(label)
    featuresVal = np.array(featuresVal)
    labelsVal = np.array(labelsVal)
    while len(featuresTest) < len(testDL.dataset):
        image, feature, label = testDL.dataset[len(featuresTest)]
        featuresTest.append(feature.numpy())
        labelsTest.append(label)
    featuresTest = np.array(featuresTest)
    labelsTest = np.array(labelsTest)
    for d in depth:
        for s in size:
            print(f"Creating forest with depth {d} and size {s}")
            RF = sklearn.ensemble.RandomForestClassifier(n_estimators=s, max_depth=d, random_state=42)
            RF.fit(featuresTrain, labelsTrain)
            ypred = RF.predict(featuresTrain)
            print("Train Accuracy:", (ypred == labelsTrain).astype(np.float32).mean())
            ypred = RF.predict(featuresVal)
            print("Validation Accuracy:", accuracy_score(labelsVal, ypred))
            ypred = RF.predict(featuresTest)
            print("Test Accuracy:", accuracy_score(labelsTest, ypred))
    return RF

class CUBMLPShort(pl.LightningModule):
    def __init__(self, input_dim=256, hidden_dim=128, output_dim=200, lr=0.001):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, output_dim),
        )
        self.nClasses = output_dim
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, concepts, labels = batch
        outputs = self(concepts)
        loss = cross_entropy(outputs, labels.argmax(dim=1))
        self.log('train/loss', loss)
        y_pred = torch.nn.functional.one_hot(outputs.argmax(dim=-1), self.nClasses).float()

        with torch.no_grad():
            task_acc = (y_pred == labels).float().mean()
            self.log("train/task_flat_acc", task_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        with torch.no_grad():
            subset_acc = ((y_pred == labels).float().prod(dim=-1)).mean()
            self.log("train/task_subset_acc", subset_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        # Most important!!!
        with torch.no_grad():
            y_true_mc = labels.argmax(dim=-1)
            y_pred_mc = y_pred.argmax(dim=-1)
            multiclass_acc = (y_true_mc == y_pred_mc).float().mean()
            self.log("train/task_multiclass_acc", multiclass_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, concepts, labels = batch
        outputs = self(concepts)
        loss = cross_entropy(outputs, labels.argmax(dim=1))
        self.log('val/loss', loss)
        y_pred = torch.nn.functional.one_hot(outputs.argmax(dim=-1), self.nClasses).float()
        with torch.no_grad():
            task_acc = (y_pred == labels).float().mean()
            self.log("val/task_flat_acc", task_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        with torch.no_grad():
            subset_acc = ((y_pred == labels).float().prod(dim=-1)).mean()
            self.log("val/task_subset_acc", subset_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        # Most important!!!
        with torch.no_grad():
            y_true_mc = labels.argmax(dim=-1)
            y_pred_mc = y_pred.argmax(dim=-1)
            multiclass_acc = (y_true_mc == y_pred_mc).float().mean()
            self.log("val/task_multiclass_acc", multiclass_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def make_perfect_string_tree(tree_str, dummy_split="0"):
    lines = [line.strip() for line in tree_str.strip().split('\n') if line.strip()]
    
    tree_lines = lines
    max_depth = len(tree_lines) - 1
    
    # --- 1. RECONSTRUCT THE TREE ---
    root = Node(tree_lines[0].strip())
    active_parents = [root]
    
    for line in tree_lines[1:]:
        tokens = [t.strip() for t in line.split('|')]
        token_idx = 0
        next_active_parents = []
        
        for parent in active_parents:
            # Assign Left Child
            left_val = tokens[token_idx]
            parent.left = Node(left_val)
            token_idx += 1
            # If it's a split, it will have children in the next line
            if not left_val.startswith("Sum="):
                next_active_parents.append(parent.left)
                
            # Assign Right Child
            right_val = tokens[token_idx]
            parent.right = Node(right_val)
            token_idx += 1
            if not right_val.startswith("Sum="):
                next_active_parents.append(parent.right)
                
        active_parents = next_active_parents

    # --- 2. PAD INTO PERFECT BINARY TREE ---
    def pad_tree(node, current_depth):
        # Base case: Reached the absolute bottom of the tree
        if current_depth == max_depth:
            return
            
        # If we hit an early leaf, turn it into a dummy split
        if node.val.startswith("Sum="):
            leaf_val = node.val
            node.val = dummy_split
            node.left = Node(leaf_val)
            node.right = Node(leaf_val)
            
        # Recurse down
        pad_tree(node.left, current_depth + 1)
        pad_tree(node.right, current_depth + 1)
        
    pad_tree(root, 0)
    
    # --- 3. EXPORT BACK TO STRING ---
    out_lines = []
    queue = [root]
    
    for d in range(max_depth + 1):
        # Extract values for the current level
        level_vals = [n.val for n in queue]
        out_lines.append(" | ".join(level_vals))
        
        # Prepare queue for the next level
        next_queue = []
        for n in queue:
            if n.left: next_queue.append(n.left)
            if n.right: next_queue.append(n.right)
        queue = next_queue
        
    return "\n".join(out_lines)


def extractTensorValues(depth: int, memorySize: int, randomState: int, nConcepts:int, nOutputs: int):
    RF = sklearn.ensemble.RandomForestClassifier(n_estimators=memorySize, max_depth=depth, random_state=randomState)
    trainDl, valDL, testDL = get_awa2_dataloaders("/Users/alecvandeuren/Thesis/src/data/raw/AWA", batch_size=1, shuffle=False, num_workers=0)
    features = []
    labels = []
    while len(features) < len(trainDl.dataset):
        image, feature, label = trainDl.dataset[len(features)]
        features.append(feature.numpy())
        labels.append(label)
    features = np.array(features)
    labels = np.array(labels)
    RF.fit(features, labels.argmax(axis=1))
    def print_forest_levels(ensemble, feature_names=None):
        out = []
        for tree_idx, estimator in enumerate(ensemble.estimators_):
            out.append([])

            tree_ = estimator.tree_
            children_left = tree_.children_left
            children_right = tree_.children_right

            feature = tree_.feature
            threshold = tree_.threshold
            value = tree_.value

            # Queue: (node_id, depth_level, parent_context_string)
            queue = deque([(0, 0, "")])
            levels = {}

            while queue:
                node_id, level, parent_str = queue.popleft()

                is_leaf = children_left[node_id] == -1

                # Format the current node's string
                if is_leaf:
                    class_counts = value[node_id][0]

                    # The prediction is the index with the highest count
                    predicted_idx = np.argmax(class_counts)

                    node_str = f"Sum={predicted_idx}"
                else:
                    # Figure out the feature name (or default to F0, F1, etc. if none provided)
                    feat_idx = str(feature[node_id])
                    if feature_names is not None:
                        feat_name = feat_idx
                    else:
                        feat_name = feat_idx
                    node_str = feat_name

                # Add it to our level dictionary
                if level not in levels:
                    levels[level] = []
                levels[level].append(node_str)

                # If it's a split node, add its children to the queue
                if not is_leaf:
                    # Left child (True branch)
                    queue.append((children_left[node_id], level + 1, f" (split{node_id}True)"))
                    # Right child (False branch)
                    queue.append((children_right[node_id], level + 1, f" (split{node_id}False)"))

            # Print out the formatted levels
            for lvl in sorted(levels.keys()):
                out[-1].append(" | ".join(levels[lvl]))
            out[-1] = "\n".join(out[-1])
        return out
    
    output = print_forest_levels(RF)
    output = [make_perfect_string_tree(tree_str) for tree_str in output]

    allSplits = [] # tensor of shape (memorySize, 2**depth-1, nConcepts)
    allLeafs = [] # tensor of shape (memorySize, 2**depth, nOutputs)

    for tree_str in output:
        lines = [line.strip() for line in tree_str.strip().split('\n') if line.strip()]
        splits = []
        leafs = []
        for line in lines:
            tokens = [t.strip() for t in line.split('|')]
            for token in tokens:
                if token.startswith("Sum="):
                    leafVal = int(token.split("=")[1])

                    leaf = torch.zeros(nOutputs)
                    leaf[leafVal] = 1.0

                    leafs.append(leaf)
                else:
                    split = torch.zeros(nConcepts)
                    split[int(token)] = 1.0
                    splits.append(split)
        leafs = torch.stack(leafs)
        splits = torch.stack(splits)
        allSplits.append(splits)
        allLeafs.append(leafs)
    allSplits = torch.stack(allSplits)
    allLeafs = torch.stack(allLeafs)
    return allSplits, allLeafs


if __name__ == "__main__":
    createForest()