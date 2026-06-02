import torch
import pytorch_lightning as pl

class AWAMLP(pl.LightningModule):
    def __init__(self, encoder, embedding_size=2048, num_classes=50):
        super(AWAMLP, self).__init__()
        self.encoder = encoder
        #self.fc1 = torch.nn.Linear(embedding_size, embedding_size)
        #self.fc2 = torch.nn.Linear(embedding_size, num_classes)
        self.fc3 = torch.nn.Linear(embedding_size, num_classes)
        self.loss = torch.nn.CrossEntropyLoss()
    

    def forward(self, x):
        x = self.encoder(x)
        #x = torch.relu(self.fc1(x))
        #x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

    def training_step(self, batch, batch_idx):
        x, c, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y.argmax(dim=-1))
        self.log('train/loss', loss)
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == y.argmax(dim=-1)).float().mean()
        self.log('train/acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, c, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y.argmax(dim=-1))
        self.log('val/loss', loss)
        # accuracy
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == y.argmax(dim=-1)).float().mean()
        self.log('val/acc', acc, prog_bar=True, on_step=False, on_epoch=True)
    

    def test_step(self, batch, batch_idx):
        x, c, y = batch
        logits = self.forward(x)
        # accuracy
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == y.argmax(dim=-1)).float().mean()
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
            },
        }