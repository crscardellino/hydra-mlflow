"""
Model module for the MLFlow Hydra Experimentation Framework

    MLFlow Hydra Experimentation Framework
    Copyright (C) 2024 Cristian Cardellino

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from lightning import LightningModule


class MultiLayerPerceptron(LightningModule):
    """
    Multi-Layer Perceptron Pytorch Lightning Module

    Parameters
    ----------
        input_size: int
            Dimension of the input.
        output_size: int
            Dimension of the output.
        layers: list[int]
            Sizes of the hidden layers (an empty list makes a linear model)
        learning_rate: float
            Learning rate parameter
        l2_lambda: float
            Regularization parameter (i.e. weight decay in the Adam optimizer)
        activation: nn.Module | None
            Activation function to use in the hidden layers.
    """
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 layers: list[int] = [64],
                 learning_rate: float = 1e-3,
                 l2_lambda: float = 1e-5,
                 activation: nn.Module | None = nn.ReLU):
        super().__init__()

        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda

        model = []

        for layer_idx, (layer_in, layer_out) in\
                enumerate(zip([input_size] + layers, layers + [output_size])):
            model.append(nn.Linear(layer_in, layer_out))
            if layer_idx < len(layers) and activation is not None:
                # Only apply activation if this isn't the last layer
                model.append(activation())

        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def training_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch

        logits = self(x)
        loss = F.cross_entropy(logits, y)

        self.log('mlp__train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            logits = self(x)
            loss = F.cross_entropy(logits, y)
            self.log('mlp__validation_loss', loss, on_epoch=True)

    def predict_step(self, batch, batch_idx):
        if not torch.is_tensor(batch):
            # Batch is a list, take first element
            batch = batch[0]
        return F.softmax(self(batch), dim=-1)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.l2_lambda
        )
