import os
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# Add project root for local imports.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.dataset import FullSpectrumDataset
from src.core.full_spectrum_models import SpectralPredictor, SpectrumGenerator


def center_of_mass(spectra: torch.Tensor) -> torch.Tensor:
    """Compute center-of-mass along spectral axis for physics-informed generator regularization."""
    seq_len = spectra.size(2)
    w = torch.arange(seq_len, device=spectra.device, dtype=torch.float32).view(1, 1, -1)
    mass = torch.sum(spectra, dim=2, keepdim=True) + 1e-8
    com = torch.sum(spectra * w, dim=2, keepdim=True) / mass
    return com.squeeze()


def train() -> None:
    device = torch.device('cpu')
    print(f'Using device: {device}')

    data_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '..',
            'data',
            'processed',
            'All_Absorbance_Spectra_Preprocessed.xlsx',
        )
    )
    print(f'Loading dataset: {os.path.basename(data_path)}')

    dataset = FullSpectrumDataset(data_path, phase='Ag')

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    seq_len = len(dataset.wavelengths)

    predictor = SpectralPredictor(seq_len=seq_len).to(device)
    generator = SpectrumGenerator(seq_len=seq_len).to(device)

    criterion_huber = nn.HuberLoss(delta=1.0)
    criterion_mse = nn.MSELoss()

    optimizer_p = optim.Adam(predictor.parameters(), lr=0.001, weight_decay=1e-4)
    optimizer_g = optim.Adam(generator.parameters(), lr=0.001)

    scheduler_p = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_p, mode='min', factor=0.5, patience=5, verbose=True
    )
    scheduler_g = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_g, mode='min', factor=0.5, patience=5, verbose=False
    )

    epochs = 150
    history = {'train_loss_p': [], 'train_loss_g': [], 'val_loss_p': [], 'val_loss_g': []}

    print(f'Start training... epochs={epochs}')
    for epoch in range(epochs):
        predictor.train()
        generator.train()

        running_loss_p = 0.0
        running_loss_g = 0.0

        for spectra, concs in train_loader:
            spectra, concs = spectra.to(device), concs.to(device)

            optimizer_p.zero_grad()
            pred_concs = predictor(spectra)
            loss_p = criterion_huber(pred_concs, concs)
            loss_p.backward()
            optimizer_p.step()
            running_loss_p += loss_p.item() * spectra.size(0)

            optimizer_g.zero_grad()
            gen_spectra = generator(concs)
            loss_mse = criterion_mse(gen_spectra, spectra)

            com_gen = center_of_mass(gen_spectra)
            com_real = center_of_mass(spectra)
            loss_shift = torch.mean((com_gen - com_real) ** 2) / 1000.0

            loss_g = loss_mse + loss_shift
            loss_g.backward()
            optimizer_g.step()
            running_loss_g += loss_g.item() * spectra.size(0)

        epoch_loss_p = running_loss_p / train_size
        epoch_loss_g = running_loss_g / train_size
        history['train_loss_p'].append(epoch_loss_p)
        history['train_loss_g'].append(epoch_loss_g)

        predictor.eval()
        generator.eval()
        val_loss_p = 0.0
        val_loss_g = 0.0
        with torch.no_grad():
            for spectra, concs in val_loader:
                spectra, concs = spectra.to(device), concs.to(device)

                pred_concs = predictor(spectra)
                val_loss_p += criterion_mse(pred_concs, concs).item() * spectra.size(0)

                gen_spectra_val = generator(concs)
                loss_mse_val = criterion_mse(gen_spectra_val, spectra)
                com_gen_val = center_of_mass(gen_spectra_val)
                com_real_val = center_of_mass(spectra)
                loss_shift_val = torch.mean((com_gen_val - com_real_val) ** 2) / 1000.0

                val_loss_g += (loss_mse_val + loss_shift_val).item() * spectra.size(0)

        history['val_loss_p'].append(val_loss_p / val_size)
        history['val_loss_g'].append(val_loss_g / val_size)

        scheduler_p.step(history['val_loss_p'][-1])
        scheduler_g.step(history['val_loss_g'][-1])

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{epochs}] | "
                f"Train(P): {epoch_loss_p:.4f}, Train(G): {epoch_loss_g:.4f} | "
                f"Val(P): {history['val_loss_p'][-1]:.4f}, Val(G): {history['val_loss_g'][-1]:.4f}"
            )

    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    os.makedirs(model_dir, exist_ok=True)

    torch.save(predictor.state_dict(), os.path.join(model_dir, 'spectral_predictor.pth'))
    torch.save(generator.state_dict(), os.path.join(model_dir, 'spectral_generator.pth'))

    norm_params = {
        'spec_min': dataset.spec_min,
        'spec_max': dataset.spec_max,
        'wavelengths': dataset.wavelengths,
    }
    torch.save(norm_params, os.path.join(model_dir, 'norm_params.pth'))
    print('Saved models and normalization params to models/.')

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss_p'], label='Train Loss')
    plt.plot(history['val_loss_p'], label='Val Loss')
    plt.title('Predictor Convergence')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss_g'], label='Train Loss', color='orange')
    plt.plot(history['val_loss_g'], label='Val Loss', color='red')
    plt.title('Generator Convergence')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()

    plt.tight_layout()
    plots_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs'))
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'training_convergence.png'), dpi=300)
    print('Saved training curve to outputs/training_convergence.png')


if __name__ == '__main__':
    train()
