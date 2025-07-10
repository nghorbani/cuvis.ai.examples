import torch
from torch import nn
import os
import sys
import argparse
import yaml
from skorch import NeuralNetRegressor
from PerPixelAECuvisDataSet import PerPixelAECuvisDataSet
# from PerPixelAEModels import HybridLoss, CosSpectralAngleLoss, Autoencoder, AutoencoderSmall
from skorch.callbacks import EarlyStopping, Checkpoint, ProgressBar
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from skorch.callbacks import TensorBoard


# Initialize the model and the Skorch wrapper
def create_skorch_model(
        device: str='cuda',
        encoding_dim: int=1,
        wave: int=6,
        large: bool=True,
        loss: str='MSE',
        use_tensorboard: bool=True,
        max_epochs: int=100,
        lr: float=0.01,
        batch_size: int=7,
    )-> NeuralNetRegressor:
    # Create early stopping patience
    early_stopping = EarlyStopping(monitor='valid_loss', patience = 10, threshold = 0.01, threshold_mode='rel', lower_is_better=True)
    if large:
        autoencoder = Autoencoder(encoding_dim, wave)
    else:
        autoencoder = AutoencoderSmall(encoding_dim, wave)
    # Choose loss function
    loss_lookup = {
        'MSE': nn.MSELoss,
        'SAM': CosSpectralAngleLoss,
        'Hybrid': HybridLoss
    }
    try:
        loss_fnc = loss_lookup[loss]
    except KeyError as e:
        print('Invalid loss function!')
        sys.exit(1)
    print(f'Using {loss} as loss function!')
    # Add a monitor to save the best performance
    monitor = lambda net: all(net.history[-1, ('train_loss_best', 'valid_loss_best')])
    checkpoint = Checkpoint(monitor=monitor, f_params="./runs/params_{last_epoch[epoch]}.pt")
    # Create initial callbacks
    callbacks = [ProgressBar(),early_stopping, checkpoint]
    # Enable us to view model performance through the TensorBoard GUI
    if use_tensorboard:
        writer = SummaryWriter()
        callbacks.append(TensorBoard(writer))
    skorch_model = NeuralNetRegressor(
        autoencoder,
        max_epochs=max_epochs,    # adjust as needed
        lr=lr,
        optimizer = torch.optim.Adam,
        criterion = loss_fnc, # Chose loss function class here
        callbacks = callbacks,
        batch_size = batch_size, # This will need to vary based on the computer utilized
        iterator_train__shuffle=True,
        device=device
    )
    return skorch_model

# Define the PyTorch model
class Autoencoder(nn.Module):
    def __init__(self, encoding_dim: int, wave: int):
        super(Autoencoder, self).__init__()
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(wave, 150),
            nn.ReLU(),
            nn.Linear(150, 100),
            nn.ReLU(),
            nn.Linear(100, 75),
            nn.ReLU(),
            nn.Linear(75, 50),
            nn.ReLU(),
            nn.Linear(50, encoding_dim)
        )
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 75),
            nn.ReLU(),
            nn.Linear(75, 100),
            nn.ReLU(),
            nn.Linear(100, wave),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class AutoencoderSmall(nn.Module):
    def __init__(self, encoding_dim: int, wave: int):
        super(AutoencoderSmall, self).__init__()
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(wave, 5),
            nn.ReLU(),
            nn.Linear(5, 3),
            nn.ReLU(),
            nn.Linear(3, encoding_dim)
        )
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 3),
            nn.ReLU(),
            nn.Linear(3, 5),
            nn.ReLU(),
            nn.Linear(5, wave),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class CosSpectralAngleLoss(nn.Module):
    def __init__(self):
        super(CosSpectralAngleLoss, self).__init__()
    def forward(self, y, y_reconstructed):
        # Normalize y and y_reconstructed along the feature dimension
        # This should normalize along the length of the spectral vector
        epsilon = 1e-8
        normalize_r = torch.sqrt(torch.sum(y_reconstructed**2, dim=1)) + epsilon
        # This should also normalize along the length of the spectral vector
        normalize_t = torch.sqrt(torch.sum(y**2, dim=1))
        # Compute cosine similarity between the normalized vectors
        numerator = torch.sum((y * y_reconstructed), dim=1)
        denominator = normalize_r * normalize_t
        cosine_similarity = numerator / denominator
        # Compute the spectral angle in radians
        spectral_angle = torch.acos(cosine_similarity)
        spectral_angle = torch.nan_to_num(cosine_similarity, 1)
        # # When a function perfectly matches, the value is 0
        # # Torch acos is define over [-1,1]
        # # To make this an appropriate loss value, we need to invert the spread and then add 1
        spectral_angle = (-1 * spectral_angle) + 1
        # Average the spectral angle to get the final loss
        loss = torch.mean(spectral_angle)
        return loss

class HybridLoss(nn.Module):
    def __init__(self):
        super(HybridLoss, self).__init__()
        self.MSELoss = nn.MSELoss()
        # Controls the weighting of the importance between MSE and SAM
        self.alpha = 0.1

    def forward(self, y, y_reconstructed):
        return self.forward_mse(y,y_reconstructed) + self.alpha * self.forward_sam(y, y_reconstructed)

    def forward_sam(self, y, y_reconstructed):
        # Normalize y and y_reconstructed along the feature dimension
        # This should normalize along the length of the spectral vector
        epsilon = 1e-8
        normalize_r = torch.sqrt(torch.sum(y_reconstructed**2, dim=1)) + epsilon
        # This should also normalize along the length of the spectral vector
        normalize_t = torch.sqrt(torch.sum(y**2, dim=1))
        # Compute cosine similarity between the normalized vectors
        # This is computer
        numerator = torch.sum((y * y_reconstructed), dim=1)
        denominator = normalize_r * normalize_t
        cosine_similarity = numerator / denominator
        # Compute the spectral angle in radians
        spectral_angle = torch.acos(cosine_similarity)
        spectral_angle = torch.nan_to_num(cosine_similarity, 1)
        # When a function perfectly matches, the value is 0
        # Torch acos is define over [-1,1]
        # To make this an appropriate loss value, we need to invert the spread and then add 1
        spectral_angle = (-1 * spectral_angle) + 1
        # Average the spectral angle to get the final loss
        loss = torch.mean(spectral_angle)
        return loss
    
    def forward_mse(self, y, y_reconstructed):
        return self.MSELoss.forward(y, y_reconstructed)
