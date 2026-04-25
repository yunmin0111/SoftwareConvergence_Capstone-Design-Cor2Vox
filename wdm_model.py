"""
WDM version of BrownianBridgeModel_c2v.
Handles 8-channel wavelet input/output instead of 1-channel.

Key changes from original:
  - channels: 8 (wavelet) instead of 1
  - UNet in_channels: 20 (x_t:8 + y:8 + condition:4)
  - UNet out_channels: 8 (wavelet MRI)
  - Brownian Bridge operates in wavelet space

Place in Cor2Vox/model/BrownianBridge/ folder.
"""

import torch
import torch.nn as nn
import numpy as np

from model.BrownianBridge.base.modules.diffusionmodules.openaimodel import UNetModel


class WdmBrownianBridgeModel(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        
        self.num_timesteps = model_config.num_timesteps
        self.channels = model_config.channels  # 8 for wavelet
        self.condition_channels = model_config.condition_channels  # 4
        self.loss_type = model_config.loss_type
        self.eta = model_config.eta
        self.max_var = model_config.max_var
        self.skip_sample = model_config.skip_sample
        self.sample_type = model_config.sample_type
        self.sample_step = model_config.sample_step
        
        # mt schedule
        self.mt_type = model_config.mt_type
        self.register_schedule()
        
        # UNet
        # in_channels = x_t(8) + y(8) + condition(4) = 20
        self.denoise_fn = UNetModel(**vars(model_config.UNetParams))
    
    def register_schedule(self):
        T = self.num_timesteps
        
        if self.mt_type == 'linear':
            m_min, m_max = 0.001, 0.999
            m_t = np.linspace(m_min, m_max, T + 1)
        elif self.mt_type == 'sin':
            m_t = np.sin(np.linspace(0, np.pi / 2, T + 1))
        else:
            raise NotImplementedError
        
        m_tminus = np.append(0, m_t[:-1])
        
        variance_t = 2. * (m_t - m_t ** 2) * self.max_var
        variance_tminus = np.append(0., variance_t[1:] * (
            (m_t[:-1] - m_tminus[:-1]) / (m_t[1:] - m_tminus[:-1])) ** 2)
        posterior_variance_t = variance_t - variance_tminus
        
        to_torch = lambda x: torch.tensor(x, dtype=torch.float32)
        self.register_buffer('m_t', to_torch(m_t))
        self.register_buffer('variance_t', to_torch(variance_t))
        self.register_buffer('variance_tminus', to_torch(variance_tminus))
        self.register_buffer('posterior_variance_t', to_torch(posterior_variance_t))
    
    def apply_model(self, x_t, t, y, condition=None):
        """Run UNet with concatenated inputs."""
        # Concatenate: x_t(8) + y(8) + condition(4) = 20ch
        context = y
        if condition is not None:
            context = torch.cat((context, condition), dim=1)
        
        x_input = torch.cat((x_t, context), dim=1)
        return self.denoise_fn(x_input, timesteps=t, context=None)
    
    def forward(self, x, y, condition=None):
        """
        Forward pass for training.
        
        Args:
            x: MRI in wavelet space (B, 8, 128, 128, 128)
            y: Sc in wavelet space (B, 8, 128, 128, 128)
            condition: conditions LLL (B, 4, 128, 128, 128)
        """
        b = x.shape[0]
        device = x.device
        
        # Random timestep
        t = torch.randint(1, self.num_timesteps + 1, (b,), device=device).long()
        
        return self.p_losses(x, y, condition=condition, t=t)
    
    def p_losses(self, x_0, y, condition=None, t=None):
        """Compute training loss in wavelet space."""
        b = x_0.shape[0]
        device = x_0.device
        
        # Get schedule values
        m_t = self.m_t[t].view(b, 1, 1, 1, 1)
        variance_t = self.variance_t[t].view(b, 1, 1, 1, 1)
        
        # Random noise (same shape as wavelet MRI: 8ch)
        noise = torch.randn_like(x_0)
        
        # Forward process: x_t = (1-m_t)*x_0 + m_t*y + sqrt(var_t)*noise
        x_t = (1. - m_t) * x_0 + m_t * y + torch.sqrt(variance_t) * noise
        
        # UNet prediction
        objective_recon = self.apply_model(x_t, t, y, condition=condition)
        
        # Target: m_t*(y - x_0) + sqrt(var_t)*noise
        objective = m_t * (y - x_0) + torch.sqrt(variance_t) * noise
        
        # Loss
        if self.loss_type == 'l1':
            recloss = (objective - objective_recon).abs().mean()
        elif self.loss_type == 'l2':
            recloss = ((objective - objective_recon) ** 2).mean()
        else:
            raise NotImplementedError
        
        return recloss
    
    @torch.no_grad()
    def p_sample_loop(self, y, condition=None):
        """
        Reverse sampling: generate MRI from SDF (both in wavelet space).
        
        Args:
            y: Sc in wavelet space (B, 8, 128, 128, 128)
            condition: conditions LLL (B, 4, 128, 128, 128)
        
        Returns:
            x_0: generated MRI in wavelet space (B, 8, 128, 128, 128)
        """
        device = y.device
        b = y.shape[0]
        
        # Start from y (shape prior in wavelet space)
        x_t = y.clone()
        
        if self.skip_sample:
            if self.sample_type == 'linear':
                timesteps = list(np.linspace(
                    1, self.num_timesteps, self.sample_step, dtype=int))
            else:
                timesteps = list(range(1, self.num_timesteps + 1))
            timesteps = list(reversed(timesteps))
        else:
            timesteps = list(reversed(range(1, self.num_timesteps + 1)))
        
        for t_idx in timesteps:
            t = torch.full((b,), t_idx, device=device, dtype=torch.long)
            
            m_t = self.m_t[t_idx]
            m_tminus = self.m_t[t_idx - 1] if t_idx > 1 else 0.
            variance_t = self.variance_t[t_idx]
            variance_tminus = self.variance_tminus[t_idx]
            posterior_variance_t = self.posterior_variance_t[t_idx]
            
            # UNet predicts the objective
            objective_recon = self.apply_model(x_t, t, y, condition=condition)
            
            # Compute x_{t-1}
            if t_idx > 1:
                noise = torch.randn_like(x_t)
            else:
                noise = torch.zeros_like(x_t)
            
            # Reverse step
            x_0_recon = (x_t - m_t * y - objective_recon) / (1. - m_t + 1e-8) + y
            # Actually use the standard BBDM reverse formula
            c_xt = (1. - m_tminus) * variance_t / (variance_t + 1e-8)
            c_yt = m_tminus * variance_t / (variance_t + 1e-8)  
            c_epst = (1. - m_tminus) / (1. - m_t + 1e-8)
            
            mean = c_xt * x_t + (m_tminus - m_t * c_xt) * y - c_epst * objective_recon * (1. - m_t)
            
            # Simplified: direct formula
            x_tminus_mean = x_t - objective_recon
            if t_idx > 1:
                x_t = x_tminus_mean + torch.sqrt(posterior_variance_t) * noise
            else:
                x_t = x_tminus_mean
        
        return x_t
