"""
Autoencoder building blocks (PyTorch).
"""


import torch
from torch import nn
import torch.nn.functional as F


class DAEEncoder(nn.Module):
    def __init__(self, input_dim, intermediate_dim=600, latent_dim=200, dropout_rate=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.dense_proj = nn.Linear(input_dim, intermediate_dim)
        self.dense_mean = nn.Linear(intermediate_dim, latent_dim)

    def forward(self, inputs):
        x = F.normalize(inputs, p=2, dim=1)
        x = self.dropout(x)
        x = torch.tanh(self.dense_proj(x))
        z = torch.tanh(self.dense_mean(x))
        return z


class DAEDecoder(nn.Module):
    def __init__(self, output_dim, intermediate_dim=600, latent_dim=200):
        super().__init__()
        self.dense_proj = nn.Linear(latent_dim, intermediate_dim)
        self.dense_output = nn.Linear(intermediate_dim, output_dim)

    def forward(self, inputs):
        x = torch.tanh(self.dense_proj(inputs))
        return self.dense_output(x)


class DenoisingAutoEncoder(nn.Module):
    def __init__(self, original_dim, intermediate_dim=600, latent_dim=200, dropout_rate=0.0):
        super().__init__()
        self.encoder = DAEEncoder(
            input_dim=original_dim,
            intermediate_dim=intermediate_dim,
            latent_dim=latent_dim,
            dropout_rate=dropout_rate,
        )
        self.decoder = DAEDecoder(
            output_dim=original_dim,
            intermediate_dim=intermediate_dim,
            latent_dim=latent_dim,
        )

    def forward(self, inputs):
        z = self.encoder(inputs)
        return self.decoder(z)


class VAEEncoder(nn.Module):
    def __init__(self, input_dim, intermediate_dim=600, latent_dim=200, dropout_rate=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.dense_proj = nn.Linear(input_dim, intermediate_dim)
        self.dense_mean = nn.Linear(intermediate_dim, latent_dim)
        self.dense_log_var = nn.Linear(intermediate_dim, latent_dim)

    def forward(self, inputs):
        x = F.normalize(inputs, p=2, dim=1)
        x = self.dropout(x)
        x = torch.tanh(self.dense_proj(x))
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        return z_mean, z_log_var


class VAEDecoder(nn.Module):
    def __init__(self, output_dim, intermediate_dim=600, latent_dim=200):
        super().__init__()
        self.dense_proj = nn.Linear(latent_dim, intermediate_dim)
        self.dense_output = nn.Linear(intermediate_dim, output_dim)

    def forward(self, inputs):
        x = torch.tanh(self.dense_proj(inputs))
        return self.dense_output(x)


class VariationalAutoEncoder(nn.Module):
    def __init__(self, original_dim, intermediate_dim=600, latent_dim=200, dropout_rate=0.0):
        super().__init__()
        self.encoder = VAEEncoder(
            input_dim=original_dim,
            intermediate_dim=intermediate_dim,
            latent_dim=latent_dim,
            dropout_rate=dropout_rate,
        )
        self.decoder = VAEDecoder(
            output_dim=original_dim,
            intermediate_dim=intermediate_dim,
            latent_dim=latent_dim,
        )

    @staticmethod
    def reparameterize(mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, inputs):
        mu, log_var = self.encoder(inputs)
        z = self.reparameterize(mu, log_var)
        logits = self.decoder(z)
        return logits, mu, log_var
