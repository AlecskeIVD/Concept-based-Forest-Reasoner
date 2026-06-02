import sklearn
from sklearn.metrics import accuracy_score
from sklearn.tree._tree import TREE_LEAF
import torch
from src.MNIST.mnist_dataset import addition_dataset
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.functional as F
import numpy as np
from collections import deque

def createDataloaders(batchSize=1):
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

    return trainDl, valDl

def extractForestSplits():
    RF = sklearn.ensemble.RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    trainDl, valDl = createDataloaders(batchSize=1)
    features = []
    labels = []
    while len(features) < len(trainDl.dataset):
        image, feature, label = trainDl.dataset[len(features)]
        features.append(feature.numpy())
        labels.append(label)
    features = np.array(features)
    labels = np.array(labels)
    RF.fit(features, labels.argmax(axis=1))
    print("Training accuracy:", accuracy_score(labels.argmax(axis=1), RF.predict(features)))
    translation = {0: "L0", 1: "L1", 2: "L2", 3: "L3", 4: "L4", 5: "L5", 6: "L6", 7: "L7", 8: "L8", 9: "L9",
                   10: "R0", 11: "R1", 12: "R2", 13: "R3", 14: "R4", 15: "R5", 16: "R6", 17: "R7", 18: "R8", 19: "R9"}
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
    output = print_forest_levels(RF, translation)

    return output

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
    trainDl, valDl = createDataloaders(batchSize=1)
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


def createForest():
    depth = [5, 8, 10]
    size = [50, 100, 300]
    trainDl, testDL = createDataloaders(batchSize=1)
    featuresTrain = []
    labelsTrain = []
    while len(featuresTrain) < len(trainDl.dataset):
        image, feature, label = trainDl.dataset[len(featuresTrain)]
        featuresTrain.append(feature.numpy())
        labelsTrain.append(label)
    featuresTrain = np.array(featuresTrain)
    labelsTrain = np.array(labelsTrain)
    featuresTest = []
    labelsTest = []
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
            RF.fit(featuresTrain, labelsTrain.argmax(axis=1))
            ypredTrain = RF.predict(featuresTrain)
            print("Train Accuracy:", (ypredTrain == labelsTrain.argmax(axis=1)).astype(np.float32).mean())
            ypred = RF.predict(featuresTest)
            print("Test Accuracy:", (ypred == labelsTest.argmax(axis=1)).astype(np.float32).mean())




if __name__ == "__main__":
    #out = extractForestSplits()
    #out = extractTensorValues(depth=3, memorySize=5, randomState=42, nConcepts=20, nOutputs=19)
    #for input_str in out:
    #    print(input_str[0])
    createForest()

