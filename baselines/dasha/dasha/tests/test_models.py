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
        self.assertEqual(len(parameters), 2)
        self.assertTrue(parameters[0].grad is not None)
        accuracy = model.accuracy(features, targets)
        self.assertTrue(accuracy >= 0 - 1e-2 and accuracy <= 1.0 + 1e-2)


if __name__ == "__main__":
    unittest.main()
