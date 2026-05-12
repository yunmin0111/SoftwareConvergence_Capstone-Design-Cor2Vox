"""
3D Autoencoder for Latent Diffusion Model (LDM).

Encodes 128³ volumes into compact 32³ latent space.
The diffusion model operates in this latent space, then decodes back.

Compression: 128³ × 1ch → 32³ × 4ch (128x compression!)
  128³ = 2,097,152 values
  32³ × 4 = 131,072 values

Phase 1: Train this autoencoder to reconstruct MRI
Phase 2: Freeze autoencoder, train diffusion in latent space

No external packages required - pure PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.act = nn.SiLU()
    
    def forward(self, x):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return x + h


class Encoder3D(nn.Module):
    """
    Encodes 128³ × 1ch → 32³ × latent_dim.
    Downsamples 4x in each spatial dimension.
    """
    def __init__(self, in_channels=1, latent_dim=4, base_channels=32):
        super().__init__()
        
        self.encoder = nn.Sequential(
            # 128³ × 1 → 128³ × 32
            nn.Conv3d(in_channels, base_channels, 3, padding=1),
            nn.SiLU(),
            ResBlock3D(base_channels),
            
            # 128³ × 32 → 64³ × 64
            nn.Conv3d(base_channels, base_channels * 2, 4, stride=2, padding=1),
            nn.SiLU(),
            ResBlock3D(base_channels * 2),
            
            # 64³ × 64 → 32³ × 128
            nn.Conv3d(base_channels * 2, base_channels * 4, 4, stride=2, padding=1),
            nn.SiLU(),
            ResBlock3D(base_channels * 4),
        )
        
        # To latent: 32³ × 128 → 32³ × latent_dim*2 (mean + logvar)
        self.to_latent = nn.Conv3d(base_channels * 4, latent_dim * 2, 1)
    
    def forward(self, x):
        h = self.encoder(x)
        params = self.to_latent(h)
        mean, logvar = params.chunk(2, dim=1)
        # Reparameterization trick
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mean + std * eps
        else:
            z = mean
        return z, mean, logvar


class Decoder3D(nn.Module):
    """
    Decodes 32³ × latent_dim → 128³ × 1ch.
    Upsamples 4x in each spatial dimension.
    """
    def __init__(self, out_channels=1, latent_dim=4, base_channels=32):
        super().__init__()
        
        self.decoder = nn.Sequential(
            # 32³ × latent_dim → 32³ × 128
            nn.Conv3d(latent_dim, base_channels * 4, 3, padding=1),
            nn.SiLU(),
            ResBlock3D(base_channels * 4),
            
            # 32³ × 128 → 64³ × 64
            nn.ConvTranspose3d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1),
            nn.SiLU(),
            ResBlock3D(base_channels * 2),
            
            # 64³ × 64 → 128³ × 32
            nn.ConvTranspose3d(base_channels * 2, base_channels, 4, stride=2, padding=1),
            nn.SiLU(),
            ResBlock3D(base_channels),
            
            # 128³ × 32 → 128³ × out_channels
            nn.Conv3d(base_channels, out_channels, 3, padding=1),
        )
    
    def forward(self, z):
        return self.decoder(z)


class AutoEncoder3D(nn.Module):
    """
    3D VAE Autoencoder for LDM.
    
    Input:  128³ × 1ch (brain MRI)
    Latent: 32³ × 4ch
    Output: 128³ × 1ch (reconstructed MRI)
    
    Loss = reconstruction_loss + kl_weight * kl_loss
    """
    def __init__(self, in_channels=1, latent_dim=4, base_channels=32, kl_weight=1e-6):
        super().__init__()
        self.encoder = Encoder3D(in_channels, latent_dim, base_channels)
        self.decoder = Decoder3D(in_channels, latent_dim, base_channels)
        self.kl_weight = kl_weight
    
    def encode(self, x):
        z, mean, logvar = self.encoder(x)
        return z, mean, logvar
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z, mean, logvar = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, mean, logvar
    
    def compute_loss(self, x, x_recon, mean, logvar):
        # Reconstruction loss
        recon_loss = F.l1_loss(x_recon, x)
        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        total_loss = recon_loss + self.kl_weight * kl_loss
        return total_loss, recon_loss, kl_loss


if __name__ == '__main__':
    print("=" * 60)
    print("3D AutoEncoder Test")
    print("=" * 60)
    
    ae = AutoEncoder3D(in_channels=1, latent_dim=4, base_channels=32)
    total = sum(p.numel() for p in ae.parameters())
    enc_params = sum(p.numel() for p in ae.encoder.parameters())
    dec_params = sum(p.numel() for p in ae.decoder.parameters())
    
    print(f"Total: {total/1e6:.2f}M")
    print(f"  Encoder: {enc_params/1e6:.2f}M")
    print(f"  Decoder: {dec_params/1e6:.2f}M")
    
    x = torch.randn(1, 1, 128, 128, 128)
    with torch.no_grad():
        x_recon, mean, logvar = ae(x)
        z, _, _ = ae.encode(x)
    
    print(f"\nInput:   {x.shape}")
    print(f"Latent:  {z.shape}")
    print(f"Output:  {x_recon.shape}")
    print(f"Compression: {x.numel()} -> {z.numel()} ({x.numel()/z.numel():.1f}x)")
