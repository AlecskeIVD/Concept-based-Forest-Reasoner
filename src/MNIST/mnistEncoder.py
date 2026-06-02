import torch
import pytorch_lightning as pl

"""
Encoder die embedding produceert voor selector en concept predictor.
credits: https://github.com/daviddebot/CMR/blob/main/experiments/mnist/models_copy.py
"""
class MNISTEncoder(pl.LightningModule):
    def __init__(self, emb_size,number_digits=2):
        super().__init__()
        self.embedding_size = emb_size
        self.number_digits = number_digits

        self.concept_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 6, 5),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(6, 16, 5),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.ReLU(True),
            torch.nn.Flatten(),
            torch.nn.Linear(256, self.embedding_size),
        )
        self.tuple_embedder = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_size * self.number_digits, self.embedding_size * self.number_digits),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_size * self.number_digits, self.embedding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_size, self.embedding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_size, self.embedding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_size, self.embedding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_size, self.embedding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_size, self.embedding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_size, self.embedding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_size, self.embedding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_size, self.embedding_size),
        )

    def forward(self, batch_x):
        embeddings = []
        for i in range(batch_x.shape[1]):
            # Loop over each digit in the tuple
            x = batch_x[:, i]
            emb = self.concept_encoder(x)
            embeddings.append(emb)
        emb = torch.cat(embeddings, dim=-1)
        emb = self.tuple_embedder(emb)
        return emb