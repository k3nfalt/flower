import unittest

import torch

from dasha.models import LinearNet, NonConvexLoss

class TestSmokeLinearNetWithNonConvexLoss(unittest.TestCase):

    def test(self) -> None:
        features = torch.rand(3, 42)
        targets = torch.Tensor([1, -1, 1])
        linear_net = LinearNet(features.shape[1], num_output_features=1)
        
        logits = linear_net(features)
        loss = NonConvexLoss()(logits, targets)
        loss.backward()
        assert linear_net.fc.weight.grad is not None


if __name__ == "__main__":
    unittest.main()
