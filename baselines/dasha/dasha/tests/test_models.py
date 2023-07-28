import unittest

import torch

from dasha.models import LinearNetWithNonConvexLoss

class TestSmokeLinearNetWithNonConvexLoss(unittest.TestCase):

    def test(self) -> None:
        features = torch.rand(3, 42)
        targets = torch.Tensor([1, -1, 1])
        model = LinearNetWithNonConvexLoss(42)
        loss = model(features, targets)
        loss.backward()
        parameters = list(model.parameters())
        assert len(parameters) == 2
        assert parameters[0].grad is not None


if __name__ == "__main__":
    unittest.main()
