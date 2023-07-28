"""Define our models, and training and eval functions.

If your model is 100% off-the-shelf (e.g. directly from torchvision without requiring
modifications) you might be better off instantiating your  model directly from the Hydra
config. In this way, swapping your model for  another one can be done without changing
the python code at all
"""
import numpy as np

import torch
import torch.nn as nn


class ClassificationModel(nn.Module):
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
    
    def accuracy(self, input: torch.Tensor, target: torch.Tensor) -> float:
        raise NotImplementedError()


class LinearNet(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int, init_with_zeros: bool = False) -> None:
        super().__init__()
        self.fc = nn.Linear(num_input_features, num_output_features)
        if init_with_zeros:
            self.fc.weight.data.fill_(0.0)
            self.fc.bias.data.fill_(0.0)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        output_tensor = self.fc(input_tensor.view(input_tensor.shape[0], -1))
        return output_tensor


class NonConvexLoss(nn.Module):
    """A nonconvex loss from Tyurin A. et al., 2023 paper :

    [DASHA: Distributed Nonconvex Optimization with Communication Compression and Optimal Oracle Complexity]
    (https://openreview.net/forum?id=VA1YpcNr7ul)
    """
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert len(input.shape) == 2 and input.shape[1] == 1
        input_target = input * target
        probs = torch.sigmoid(input_target)
        loss = torch.square(1 - probs)
        loss = torch.mean(loss)
        return loss


class LinearNetWithNonConvexLoss(ClassificationModel):
    def __init__(self, num_input_features: int, init_with_zeros: bool = False) -> None:
        super().__init__()
        self._net = LinearNet(num_input_features, num_output_features=1, init_with_zeros=init_with_zeros)
        self._loss = NonConvexLoss()
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = self._net(input)
        return self._loss(logits, target)
    
    @torch.no_grad()
    def accuracy(self, input: torch.Tensor, target: torch.Tensor) -> float:
        logits = self._net(input).numpy().flatten()
        target = target.numpy()
        predictions = 2 * (logits >= 0.0) - 1
        return np.sum(predictions == target) / len(target)
