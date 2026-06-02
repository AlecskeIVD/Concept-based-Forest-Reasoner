import torch
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from src.AWA.awaDataset import getAWA2EmbeddingDataloaders
from torch.nn.functional import binary_cross_entropy_with_logits
from src.AWA.awaEncoder import AWAEncoder, AWALinearBackbone
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import logging
from itertools import product

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


class awaConceptPredictor(pl.LightningModule):
    def __init__(self, encoder, embedding_size=2048, num_classes=50):
        super(awaConceptPredictor, self).__init__()
        self.encoder = encoder
        self.fc1 = torch.nn.Linear(embedding_size, embedding_size)
        self.fc2 = torch.nn.Linear(embedding_size, num_classes)
    
    def forward(self, x):
        x = self.encoder(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, c, y = batch
        logits = self.forward(x)
        loss = binary_cross_entropy_with_logits(logits, c.float())
        self.log('train/loss', loss)
        acc = ((logits > 0) == c).float().mean()
        self.log('train/acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, c, y = batch
        logits = self.forward(x)
        loss = binary_cross_entropy_with_logits(logits, c.float())
        self.log('val/loss', loss)
        acc = ((logits > 0) == c).float().mean()
        self.log('val/acc', acc, prog_bar=True, on_step=False, on_epoch=True)
    

    def test_step(self, batch, batch_idx):
        x, c, y = batch
        logits = self.forward(x)
        acc = ((logits > 0) == c).float().mean()
        self.log('test/acc', acc)
    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val/loss" 
            }
        }
    
    def extractConceptPreds(self, dataloader):
        self.eval()
        all_preds = []

        with torch.no_grad():
            for x, c, y in dataloader:
                logits = self.forward(x.to(self.device))
                preds = (logits > 0).cpu().numpy()
                all_preds.append(preds)
        return np.vstack(all_preds)

def extract_numpy_data(dataloader):
    """
    Iterates through a PyTorch DataLoader and extracts the features and labels 
    into standard NumPy arrays for scikit-learn / XGBoost.
    """
    X_list = []
    y_list = []
    
    for embeddings, concepts, labels in dataloader:
        # We will use the embeddings as our features. 
        # (Alternatively, you could use: torch.cat((embeddings, concepts), dim=1) if you want both!)
        X_list.append(concepts.cpu().numpy())
        
        # Your dataset returns one-hot labels, but XGBoost wants integer class indices (0, 1, 2...)
        # We use argmax to convert [0, 0, 1, 0] back into index 2.
        class_indices = torch.argmax(labels, dim=-1)
        y_list.append(class_indices.cpu().numpy())
        
    # Stack all batches into one giant 2D array
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    print(f"Extracted {X.shape[0]} samples with {X.shape[1]} features each.")
    print(f"Extracted {len(y)} labels with {len(np.unique(y))} unique classes.")
    return X, y

def runXGBExperiment():
    EMB_SIZE = 256
    nConcepts = 85
    nOutputs = 50
    batchSize = 32
    depths = [7, 5, 3]
    memorySizes = [100, 50, 20]
    maxEpochs = 50
    torch.manual_seed(42) 
    train_loader, val_loader, test_loader = getAWA2EmbeddingDataloaders(
        batch_size=32, # You can use a large batch size here since no GPU/Images are needed
        embeddings_file="/Users/alecvandeuren/Thesis/src/data/raw/AWA/awa_embeddings.pt",
        root_dir="/Users/alecvandeuren/Thesis/src/data/raw/AWA"
    )
    X_train, y_train = extract_numpy_data(train_loader)

    X_val, y_val = extract_numpy_data(val_loader)

    #X_test, y_test = extract_numpy_data(test_loader)
    allAccuracies = []
    for depth, mem in product(depths, memorySizes):
        accuracies = []
        for trial in range(3):
            print(f"\nRunning XGBoost with max_depth={depth}, max_leaves={mem}, trial {trial+1}/3")
            torch.manual_seed(42 + trial)
            xgb_model = xgb.XGBClassifier(
                n_estimators=mem,      
                max_depth=depth,           
                learning_rate=0.1,     
                objective='multi:softmax', 
                tree_method='hist',    
                n_jobs=-1,
                early_stopping_rounds=15,
                random_state=42 + trial
            )

            xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)], 
                verbose=10                 
            )
            encoder = AWALinearBackbone(2048, EMB_SIZE)
            conceptPredictor = awaConceptPredictor(encoder=encoder, embedding_size=EMB_SIZE, num_classes=nConcepts)
            checkpoint_cb = ModelCheckpoint(dirpath="./model/AWA/", save_top_k=1, monitor="val/loss", mode='min')        
            trainer = pl.Trainer(max_epochs=maxEpochs, accelerator='auto', devices="auto", callbacks=[checkpoint_cb, EarlyStopping(monitor="val/loss", mode="min", patience=5)], enable_model_summary=False)
            trainer.fit(model=conceptPredictor, train_dataloaders=train_loader, val_dataloaders=val_loader)
            conceptPredictor = awaConceptPredictor.load_from_checkpoint(checkpoint_cb.best_model_path, encoder=encoder, embedding_size=EMB_SIZE, num_classes=nConcepts)
            cPreds = conceptPredictor.extractConceptPreds(val_loader)
            print(f"Extracted concept predictions shape: {cPreds.shape}")
            yPreds = xgb_model.predict(cPreds)
            acc = accuracy_score(y_val, yPreds)
            print(f"Validation Accuracy for XGBoost with max_depth={depth}, max_leaves={mem}, trial {trial+1}/3: {acc * 100:.2f}%")
            accuracies.append(acc)
        avg_acc = np.mean(accuracies)
        stdAccuracy = (sum([(x-avg_acc)**2 for x in accuracies])/len(accuracies))**0.5
        allAccuracies.append((avg_acc, stdAccuracy, (depth, mem)))
        print(f"\nAverage Validation Accuracy for max_depth={depth}, max_leaves={mem}: {avg_acc * 100:.2f}% ± {stdAccuracy * 100:.2f}%")
    print("\nSummary of all configurations:")
    allAccuracies.sort(key=lambda x: x[0]-x[1], reverse=True)
    for avg_acc, std_acc, (depth, mem) in allAccuracies:
        print(f"max_depth={depth}, max_trees={mem}: {avg_acc * 100:.2f}% ± {std_acc * 100:.2f}%")
    

"""
max_depth=5, max_trees=100: 91.28% ± 1.06%
max_depth=3, max_trees=100: 91.28% ± 1.06%
max_depth=7, max_trees=100: 90.87% ± 0.77%
max_depth=7, max_trees=50: 90.73% ± 1.21%
max_depth=5, max_trees=50: 90.73% ± 1.21%
max_depth=3, max_trees=50: 90.73% ± 1.21%
max_depth=7, max_trees=20: 90.45% ± 1.26%
max_depth=5, max_trees=20: 90.45% ± 1.26%
max_depth=3, max_trees=20: 90.45% ± 1.26%"""

def ComputeAccuracy():
    EMB_SIZE = 256
    nConcepts = 85
    nOutputs = 50
    batchSize = 32
    depths = [5]
    memorySizes = [100]
    maxEpochs = 50
    torch.manual_seed(42) 
    train_loader, val_loader, test_loader = getAWA2EmbeddingDataloaders(
        batch_size=32, # You can use a large batch size here since no GPU/Images are needed
        embeddings_file="/Users/alecvandeuren/Thesis/src/data/raw/AWA/awa_embeddings.pt",
        root_dir="/Users/alecvandeuren/Thesis/src/data/raw/AWA"
    )
    X_train, y_train = extract_numpy_data(train_loader)

    X_val, y_val = extract_numpy_data(val_loader)

    X_test, y_test = extract_numpy_data(test_loader)
    allAccuracies = []
    for depth, mem in product(depths, memorySizes):
        accuracies = []
        for trial in range(3):
            print(f"\nRunning XGBoost with max_depth={depth}, max_leaves={mem}, trial {trial+1}/3")
            torch.manual_seed(42 + trial)
            xgb_model = xgb.XGBClassifier(
                n_estimators=mem,      
                max_depth=depth,           
                learning_rate=0.1,     
                objective='multi:softmax', 
                tree_method='hist',    
                n_jobs=-1,
                early_stopping_rounds=15,
                random_state=42 + trial
            )

            xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)], 
                verbose=10                 
            )
            encoder = AWALinearBackbone(2048, EMB_SIZE)
            conceptPredictor = awaConceptPredictor(encoder=encoder, embedding_size=EMB_SIZE, num_classes=nConcepts)
            checkpoint_cb = ModelCheckpoint(dirpath="./model/AWA/", save_top_k=1, monitor="val/loss", mode='min')        
            trainer = pl.Trainer(max_epochs=maxEpochs, accelerator='auto', devices="auto", callbacks=[checkpoint_cb, EarlyStopping(monitor="val/loss", mode="min", patience=5)], enable_model_summary=False)
            trainer.fit(model=conceptPredictor, train_dataloaders=train_loader, val_dataloaders=val_loader)
            conceptPredictor = awaConceptPredictor.load_from_checkpoint(checkpoint_cb.best_model_path, encoder=encoder, embedding_size=EMB_SIZE, num_classes=nConcepts)
            cPreds = conceptPredictor.extractConceptPreds(test_loader)
            print(f"Extracted concept predictions shape: {cPreds.shape}")
            yPreds = xgb_model.predict(cPreds)
            acc = accuracy_score(y_test, yPreds)
            print(f"Test Accuracy for XGBoost with max_depth={depth}, max_leaves={mem}, trial {trial+1}/3: {acc * 100:.2f}%")
            accuracies.append(acc)
        avg_acc = np.mean(accuracies)
        stdAccuracy = (sum([(x-avg_acc)**2 for x in accuracies])/len(accuracies))**0.5
        allAccuracies.append((avg_acc, stdAccuracy, (depth, mem)))
        print(f"\nAverage Test Accuracy for max_depth={depth}, max_leaves={mem}: {avg_acc * 100:.2f}% ± {stdAccuracy * 100:.2f}%")
    print("\nSummary of all configurations:")
    allAccuracies.sort(key=lambda x: x[0]-x[1], reverse=True)
    for avg_acc, std_acc, (depth, mem) in allAccuracies:
        print(f"max_depth={depth}, max_trees={mem}: {avg_acc * 100:.2f}% ± {std_acc * 100:.2f}%")
    # Average Test Accuracy for max_depth=5, max_leaves=100: 90.50% ± 0.89%
            

