import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from composer import Trainer, ComposerModel
import torchelie.nn as tnn
from model import Transformer


def make_denoiser(vocab_size, hidden_dim, num_heads, head_size, num_layers):
    tfm = Transformer(
        vocab_size, hidden_dim, num_heads, head_size, num_layers, causal=False
    )
    tfm.vocab = nn.Identity()
    tfm.proj = nn.Identity()
    return tfm


class ComposerDenoiser(ComposerModel):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch):
        inputs, t = batch
        self.encoder(inputs, t)

    def loss(self, outputs, batch):
        inputs, t = batch
        return F.mse_loss(outputs, inputs)


if False:
    train_dataloader = DataLoader(dataset, batch_size=128)

    trainer = Trainer(
        model=ComposerClassifier(module=Model(), num_classes=10),
        train_dataloader=train_dataloader,
    )
    trainer.fit()

model = ComposerDenoiser(*make_denoiser(100, 256, 8, 32, 6, 32, 0.1))
out = model((torch.randint(0, 100, size=(3, 4)), None))
print(out.shape)
