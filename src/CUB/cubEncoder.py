import torch
import pytorch_lightning as pl
import torchvision.models as models

"""
Encoder die embedding produceert voor selector en concept predictor.
"""
class CUBEncoderOLD(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.concept_encoder = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        for param in self.concept_encoder.parameters():
            param.requires_grad = False
        self.output_dim = self.concept_encoder.fc.in_features
        self.concept_encoder.fc = torch.nn.Identity()
        

    def forward(self, batch_x):
        if self.training:
            return self.concept_encoder(batch_x)[0]
        return self.concept_encoder(batch_x)


class LinearEncoder(pl.LightningModule):
    def __init__(self, input_dim=2048, output_dim=256):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, batch_x):
        return self.linear(batch_x)


if __name__ == "__main__":
    # test the encoder with dummy data
    encoder = CUBEncoder()
    encoder.eval()
    dummy_input = torch.randn(1, 3, 299, 299)
    output = encoder(dummy_input)
    print("Output shape:", output.shape)
    for name, param in encoder.concept_encoder.named_parameters():
        if param.requires_grad:
            print(f"Training layer: {name}")


class CUBEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.concept_encoder = models.inception(weights=models.GoogLeNet_Weights.DEFAULT)
        for param in self.concept_encoder.parameters():
            param.requires_grad = False
        self.output_dim = self.concept_encoder.fc.in_features
        self.concept_encoder.fc = torch.nn.Identity()
        

    def forward(self, batch_x):
        if self.training:
            return self.concept_encoder(batch_x)[0]
        return self.concept_encoder(batch_x)
    

class CUBEmbeddingExtractor(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.concept_encoder = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        self.output_dim = self.concept_encoder.fc.in_features
        self.concept_encoder.fc = torch.nn.Identity()
        for param in self.concept_encoder.parameters():
            param.requires_grad = False

    def forward(self, batch_x):
        if self.training:
            return self.concept_encoder(batch_x)[0]
        return self.concept_encoder(batch_x)


class CUBLinearEmbeddingExtractor(pl.LightningModule):
    def __init__(self, input_dim=2048, output_dim=256):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, input_dim)
        self.linear2 = torch.nn.Linear(input_dim, output_dim)

    def forward(self, batch_x):
        x = self.linear(batch_x)
        return self.linear2(x)
