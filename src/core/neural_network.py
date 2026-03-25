import torch.nn as nn


class LSPRResidualNet(nn.Module):
    """Lightweight regressor for residual physics features (delta-lambda, delta-A)."""

    def __init__(self, input_dim: int = 4, output_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, x):
        return self.net(x)
