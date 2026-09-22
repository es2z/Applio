"""FP32 inference port of ardaillon/FCN-f0's FCN_993 (MIT)."""

import torch
from torch import nn

ARCHITECTURE = {
    "id": "fcn-993",
    "sample_rate": 8000,
    "receptive_field": 993,
    "stride": 8,
    "bins": 486,
    "fmin": 30.0,
    "fmax": 1000.0,
}


class FCNModel(nn.Module):
    def __init__(self):
        super().__init__()
        channels = (1, 256, 32, 32, 128, 256, 512)
        for i in range(1, 7):
            self.add_module(f"conv{i}", nn.Conv1d(channels[i - 1], channels[i], 32))
            self.add_module(
                f"bn{i}", nn.BatchNorm1d(channels[i], eps=0.001, momentum=0.01)
            )
        self.classifier = nn.Conv1d(512, 486, 4)
        self.requires_grad_(False)
        self.eval()

    def forward(self, x):
        """[batch, 1, samples] -> [batch, native frames, 486]."""
        for i in range(1, 7):
            x = torch.relu(getattr(self, f"conv{i}")(x))
            if i < 4:
                x = torch.nn.functional.max_pool1d(x, 2, 2)
            x = getattr(self, f"bn{i}")(x)
        return torch.sigmoid(self.classifier(x)).transpose(1, 2)
