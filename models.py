import torch
import torch.nn as nn

class SAR_VAE(nn.Module):
    """
    Variational Autoencoder (SAR-VAE) utilizing a stochastic KL-divergence 
    bottleneck to map normative clean ocean backscatter features.
    """
    def __init__(self):
        super(SAR_VAE, self).__init__()
        # Encoder: 256x256 -> 128x128 -> 64x64
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 64 * 64, 128)
        )
        self.fc_mu = nn.Linear(128, 64)
        self.fc_var = nn.Linear(128, 64)
        
        # Decoder
        self.decoder_input = nn.Linear(64, 64 * 64 * 64)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        h = torch.relu(self.encoder(x))
        mu, log_var = self.fc_mu(h), self.fc_var(h)
        
        # Reparameterization trick
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        d_in = torch.relu(self.decoder_input(z))
        d_in = d_in.view(-1, 64, 64, 64)
        return self.decoder(d_in), mu, log_var


class SAR_AAE(nn.Module):
    """
    Adversarial Autoencoder (SAR-AAE) combining an autoencoder backbone with 
    minimax adversarial regularization on the latent space.
    """
    def __init__(self):
        super(SAR_AAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 64 * 64, 64)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 64 * 64 * 64),
            nn.Unflatten(1, (64, 64, 64)),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z