"""FP32 inference ports of ardaillon/FCN-f0's FCN_993 and FCN_929 (MIT)."""

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

ARCHITECTURES = {
    "fcn-993": ARCHITECTURE,
    "fcn-929": {
        "id": "fcn-929",
        "sample_rate": 8000,
        "receptive_field": 929,
        "stride": 4,
        "bins": 486,
        "fmin": 30.0,
        "fmax": 1000.0,
    },
}

# Upstream models/FCN_*/core.py: conv filters, conv widths, and how many of the
# leading conv layers are followed by a 2x max-pool.
LAYERS = {
    "fcn-993": ((256, 32, 32, 128, 256, 512), (32, 32, 32, 32, 32, 32), 3),
    "fcn-929": ((256, 32, 128, 256, 512), (32, 64, 64, 64, 64), 2),
}


def normalization_window(architecture):
    """Upstream sliding_norm rounds an odd input size up to an even window."""
    size = ARCHITECTURES[architecture]["receptive_field"]
    return size + size % 2


class FCNModel(nn.Module):
    def __init__(self, architecture="fcn-993"):
        super().__init__()
        filters, widths, self.pooled = LAYERS[architecture]
        self.architecture = architecture
        self.depth = len(filters)
        channels = (1, *filters)
        for i in range(1, self.depth + 1):
            self.add_module(
                f"conv{i}", nn.Conv1d(channels[i - 1], channels[i], widths[i - 1])
            )
            self.add_module(
                f"bn{i}", nn.BatchNorm1d(channels[i], eps=0.001, momentum=0.01)
            )
        self.classifier = nn.Conv1d(channels[-1], 486, 4)
        self.requires_grad_(False)
        self.eval()

    def forward(self, x):
        """[batch, 1, samples] -> [batch, native frames, 486]."""
        for i in range(1, self.depth + 1):
            x = torch.relu(getattr(self, f"conv{i}")(x))
            if i <= self.pooled:
                x = torch.nn.functional.max_pool1d(x, 2, 2)
            x = getattr(self, f"bn{i}")(x)
        return torch.sigmoid(self.classifier(x)).transpose(1, 2)
