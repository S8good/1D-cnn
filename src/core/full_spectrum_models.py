import torch
import torch.nn as nn


class SpectralPredictor(nn.Module):
    """Model A: spectrum -> concentration (log10 scale)."""

    def __init__(self, seq_len):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(2),
            nn.Dropout1d(0.1),
            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(2),
            nn.Dropout1d(0.1),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(16),
        )
        self.regressor = nn.Sequential(
            nn.Linear(64 * 16, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 64),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        features = self.conv_blocks(x)
        features = features.view(features.size(0), -1)
        return self.regressor(features)


class SpectrumGenerator(nn.Module):
    """Model B: concentration -> full spectrum."""

    def __init__(self, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.fc = nn.Sequential(
            nn.Linear(1, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64 * 32),
            nn.LeakyReLU(0.2),
        )
        self.conv_blocks = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(32, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(16, 8, kernel_size=5, padding=2),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2),
            nn.Upsample(size=seq_len, mode='linear', align_corners=False),
            nn.Conv1d(8, 1, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.fc(x)
        out = out.view(out.size(0), 64, 32)
        out = self.conv_blocks(out)
        return out


class SpectralPredictorV2(nn.Module):
    """Robust predictor using raw intensity + first-derivative channels."""

    def __init__(self, seq_len):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Dropout1d(0.1),
            nn.Conv1d(32, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Dropout1d(0.1),
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(24),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 24, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.head(self.features(x))


class SpectralPredictorV2_Fusion(nn.Module):
    """Model C: robust spectral branch + BSA-physics feature fusion."""

    def __init__(self, seq_len):
        super().__init__()
        self.spectral_features = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Dropout1d(0.1),
            nn.Conv1d(32, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Dropout1d(0.1),
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(24),
        )
        self.spectral_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 24, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.GELU(),
        )
        self.physics_encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU(),
        )
        self.regressor = nn.Sequential(
            nn.Linear(128 + 32, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x_spectrum, x_physics):
        spec_embed = self.spectral_head(self.spectral_features(x_spectrum))
        phy_embed = self.physics_encoder(x_physics)
        fused = torch.cat([spec_embed, phy_embed], dim=1)
        return self.regressor(fused)
