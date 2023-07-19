"""Define our models, and training and eval functions.

If your model is 100% off-the-shelf (e.g. directly from torchvision without requiring
modifications) you might be better off instantiating your  model directly from the Hydra
config. In this way, swapping your model for  another one can be done without changing
the python code at all
"""

import torch
import torch.nn as nn


class LinearNet(nn.Module):
    """A simple net with one linear layer"""

    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super().__init__()
        self.fc = nn.Linear(num_input_features, num_output_features)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        output_tensor = self.fc(input_tensor.view(input_tensor.shape[0], -1))
        return output_tensor


class NonConvexLoss(nn.Module):
    """A nonconvex loss from Tyurin A. et al., 2023 paper :

    [DASHA: Distributed Nonconvex Optimization with Communication Compression and Optimal Oracle Complexity]
    (https://openreview.net/forum?id=VA1YpcNr7ul)
    """
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        input: torch.Tensor
            Logits of a prediction model
        target : torch.Tensor
            Assumes that the elements of target belong to the set {-1, 1}

        Returns
        -------
        torch.Tensor
            Loss value
        """
        assert len(input.shape) == 2 and input.shape[1] == 1
        input_target = input * target
        probs = torch.sigmoid(input_target)
        loss = torch.square(1 - probs)
        loss = torch.mean(loss)
        return loss


class LinearNetWithNonConvexLoss(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super().__init__()
        self._net = LinearNet(num_input_features, num_output_features)
        self._loss = NonConvexLoss()
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = self._net(input)
        return self._loss(logits, target)
