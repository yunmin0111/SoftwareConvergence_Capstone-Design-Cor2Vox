"""
WDM Runner for Cor2Vox.
Handles training, validation, and sampling with wavelet transform.

Key addition: IWT (Inverse Wavelet Transform) applied after sampling
to convert 128³ × 8ch wavelet back to 256³ × 1ch MRI.

Place in Cor2Vox/runners/ folder or run standalone.

Usage:
    python wdm_runner.py --config configs/c2v_wdm.yaml --gpu_ids 0 --train
    python wdm_runner.py --config configs/c2v_wdm.yaml --gpu_ids 0 --resume_model path/to/checkpoint.pth
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
import pywt
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import argparse
import copy


def iwt3d(subbands, wavelet='haar'):
    """3D IWT: (8, H, W, D) → (H*2, W*2, D*2)"""
    keys = sorted(['aaa', 'aad', 'ada', 'add', 'daa', 'dad', 'dda', 'ddd'])
    coeffs = {k: subbands[i] for i, k in enumerate(keys)}
    return pywt.idwtn(coeffs, wavelet).astype(np.float32)


class WdmRunner:
    def __init__(self, config, gpu_id=0):
        self.config = config
        self.device = torch.device(f'cuda:{gpu_id}')
        
        # Paths
        model_name = config['model']['model_name']
        self.result_dir = f'results/c2v/{model_name}'
        self.checkpoint_dir = os.path.join(self.result_dir, 'checkpoint')
        self.sample_dir = os.path.join(self.result_dir, 'samples')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)
        
        # Build model
        self._build_model()
        
        # Load checkpoint if specified
        load_path = config['model'].get('model_load_path', '')
        if load_path and os.path.exists(load_path):
            self._load_checkpoint(load_path)
    
    def _build_model(self):
        """Build WDM model."""
        from model.BrownianBridge.base.modules.diffusionmodules.openaimodel import UNetModel
        
        bb_config = self.config['model']['BB']
        params = bb_config['params']
        
        # Build UNet params
        unet_config = bb_config['UNetParams']
        
        self.model = WdmBBModel(params, unet_config).to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params / 1e6:.2f}M")
        print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
        
        # Optimizer
        opt_config = bb_config['optimizer']
        self.optimizer = Adam(
            self.model.parameters(),
            lr=opt_config['lr'],
            weight_decay=opt_config.get('weight_decay', 0)
        )
        
        # Scheduler
        sched_config = bb_config.get('lr_scheduler', {})
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            factor=sched_config.get('factor', 0.5),
            patience=sched_config.get('patience', 20),
            verbose=True
        )
        
        self.start_epoch = 0
    
    def _load_checkpoint(self, path):
        """Load model checkpoint."""
        print(f"Loading checkpoint from {path}")
        state = torch.load(path, map_location='cpu')
        if 'model' in state:
            self.model.load_state_dict(state['model'])
            if 'optimizer' in state:
                self.optimizer.load_state_dict(state['optimizer'])
            if 'epoch' in state:
                self.start_epoch = state['epoch']
            print(f"Loaded checkpoint at epoch {self.start_epoch}")
        else:
            # Try loading as raw state dict
            self.model.load_state_dict(state, strict=False)
            print("Loaded raw state dict")
    
    def _save_checkpoint(self, epoch):
        """Save checkpoint."""
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch,
            'config': self.config,
        }
        torch.save(state, os.path.join(self.checkpoint_dir, 'last_model.pth'))
        
        # Save milestone every 20 epochs
        if epoch % 20 == 0:
            torch.save(state, os.path.join(
                self.checkpoint_dir, f'model_epoch_{epoch}.pth'))
    
    def _get_dataloader(self, stage):
        """Create dataloader."""
        sys.path.insert(0, os.getcwd())
        from wdm_dataset import WdmC2vDataset
        
        dataset = WdmC2vDataset(
            self.config['data']['dataset_config'], stage=stage)
        
        loader = DataLoader(
            dataset,
            batch_size=1,  # 3D volumes are large
            shuffle=(stage == 'train'),
            num_workers=4,
            pin_memory=True,
            drop_last=(stage == 'train'),
        )
        return loader
    
    def train(self):
        """Training loop."""
        train_loader = self._get_dataloader('train')
        val_loader = self._get_dataloader('val')
        
        max_epochs = 400
        print(f"Start training from epoch {self.start_epoch}, "
              f"{len(train_loader)} iters per epoch")
        
        for epoch in range(self.start_epoch, max_epochs):
            self.model.train()
            epoch_loss = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{max_epochs}]")
            for batch_idx, batch in enumerate(pbar):
                x = batch['x'].to(self.device)
                y = batch['y'].to(self.device)
                condition = batch['condition'].to(self.device)
                
                self.optimizer.zero_grad()
                loss = self.model(x, y, condition=condition)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch+1}: avg_loss={avg_loss:.4f}")
            
            # Validation
            if (epoch + 1) % 5 == 0:
                val_loss = self._validate(val_loader)
                self.scheduler.step(val_loss)
                print(f"  val_loss={val_loss:.4f}")
            
            # Save checkpoint
            self._save_checkpoint(epoch + 1)
    
    def _validate(self, val_loader):
        """Run validation."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].to(self.device)
                y = batch['y'].to(self.device)
                condition = batch['condition'].to(self.device)
                loss = self.model(x, y, condition=condition)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def sample(self, num_samples=5):
        """Generate samples and convert back to 256³ via IWT."""
        test_loader = self._get_dataloader('test')
        self.model.eval()
        
        print(f"Sampling {num_samples} volumes...")
        
        for i, batch in enumerate(test_loader):
            if i >= num_samples:
                break
            
            y = batch['y'].to(self.device)
            condition = batch['condition'].to(self.device)
            x_real = batch['x']  # keep on CPU
            
            # Generate in wavelet space
            with torch.no_grad():
                x_syn_wavelet = self.model.p_sample_loop(
                    y, condition=condition)
            
            # Convert wavelet → 256³ via IWT
            syn_wavelet_np = x_syn_wavelet[0].cpu().numpy()  # (8, 128, 128, 128)
            syn_256 = iwt3d(syn_wavelet_np)  # (256, 256, 256)
            
            real_wavelet_np = x_real[0].numpy()  # (8, 128, 128, 128)
            real_256 = iwt3d(real_wavelet_np)  # (256, 256, 256)
            
            # Save
            nib.save(
                nib.Nifti1Image(syn_256, affine=np.eye(4)),
                os.path.join(self.sample_dir, f'sample{i}_syn_256.nii.gz'))
            nib.save(
                nib.Nifti1Image(real_256, affine=np.eye(4)),
                os.path.join(self.sample_dir, f'sample{i}_real_256.nii.gz'))
            
            print(f"  Sample {i}: syn {syn_256.shape} [{syn_256.min():.4f}, {syn_256.max():.4f}]"
                  f" | real {real_256.shape} [{real_256.min():.4f}, {real_256.max():.4f}]")
        
        print(f"Samples saved to {self.sample_dir}")


class WdmBBModel(nn.Module):
    """Brownian Bridge model operating in wavelet space."""
    
    def __init__(self, params, unet_config):
        super().__init__()
        
        self.num_timesteps = params['num_timesteps']
        self.channels = params['channels']
        self.loss_type = params['loss_type']
        self.eta = params['eta']
        self.max_var = params['max_var']
        self.skip_sample = params['skip_sample']
        self.sample_type = params['sample_type']
        self.sample_step = params['sample_step']
        self.mt_type = params['mt_type']
        
        self.register_schedule()
        
        from model.BrownianBridge.base.modules.diffusionmodules.openaimodel import UNetModel
        self.denoise_fn = UNetModel(**unet_config)
    
    def register_schedule(self):
        T = self.num_timesteps
        if self.mt_type == 'linear':
            m_min, m_max = 0.001, 0.999
            m_t = np.linspace(m_min, m_max, T + 1)
        else:
            m_t = np.sin(np.linspace(0, np.pi / 2, T + 1))
        
        m_tminus = np.append(0, m_t[:-1])
        variance_t = 2. * (m_t - m_t ** 2) * self.max_var
        variance_tminus = np.append(0., variance_t[1:] * (
            (m_t[:-1] - m_tminus[:-1]) / (m_t[1:] - m_tminus[:-1] + 1e-8)) ** 2)
        posterior_variance_t = variance_t - variance_tminus
        
        to_torch = lambda x: torch.tensor(x, dtype=torch.float32)
        self.register_buffer('m_t', to_torch(m_t))
        self.register_buffer('variance_t', to_torch(variance_t))
        self.register_buffer('variance_tminus', to_torch(variance_tminus))
        self.register_buffer('posterior_variance_t', to_torch(posterior_variance_t))
    
    def forward(self, x, y, condition=None):
        b = x.shape[0]
        device = x.device
        t = torch.randint(1, self.num_timesteps + 1, (b,), device=device).long()
        return self.p_losses(x, y, condition=condition, t=t)
    
    def p_losses(self, x_0, y, condition=None, t=None):
        b = x_0.shape[0]
        device = x_0.device
        
        m_t = self.m_t[t].view(b, 1, 1, 1, 1)
        variance_t = self.variance_t[t].view(b, 1, 1, 1, 1)
        
        noise = torch.randn_like(x_0)
        x_t = (1. - m_t) * x_0 + m_t * y + torch.sqrt(variance_t) * noise
        
        # UNet input: concat x_t + y + condition
        context = y
        if condition is not None:
            context = torch.cat((context, condition), dim=1)
        unet_input = torch.cat((x_t, context), dim=1)
        
        objective_recon = self.denoise_fn(unet_input, timesteps=t, context=None)
        objective = m_t * (y - x_0) + torch.sqrt(variance_t) * noise
        
        if self.loss_type == 'l1':
            return (objective - objective_recon).abs().mean()
        else:
            return ((objective - objective_recon) ** 2).mean()
    
    @torch.no_grad()
    def p_sample_loop(self, y, condition=None):
        device = y.device
        b = y.shape[0]
        x_t = y.clone()
        
        if self.skip_sample and self.sample_type == 'linear':
            timesteps = list(np.linspace(
                1, self.num_timesteps, self.sample_step, dtype=int))
        else:
            timesteps = list(range(1, self.num_timesteps + 1))
        timesteps = list(reversed(timesteps))
        
        for t_idx in tqdm(timesteps, desc="sampling loop time step"):
            t = torch.full((b,), t_idx, device=device, dtype=torch.long)
            
            m_t = self.m_t[t_idx]
            variance_t = self.variance_t[t_idx]
            posterior_var = self.posterior_variance_t[t_idx]
            
            context = y
            if condition is not None:
                context = torch.cat((context, condition), dim=1)
            unet_input = torch.cat((x_t, context), dim=1)
            
            objective_recon = self.denoise_fn(unet_input, timesteps=t, context=None)
            
            x_tminus_mean = x_t - objective_recon
            
            if t_idx > 1:
                noise = torch.randn_like(x_t)
                x_t = x_tminus_mean + torch.sqrt(posterior_var) * noise
            else:
                x_t = x_tminus_mean
        
        return x_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--resume_model', type=str, default='')
    parser.add_argument('--num_samples', type=int, default=5)
    args = parser.parse_args()
    
    gpu_id = int(args.gpu_ids.split(',')[0])
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if args.resume_model:
        config['model']['model_load_path'] = args.resume_model
    
    runner = WdmRunner(config, gpu_id=gpu_id)
    
    if args.train:
        runner.train()
    else:
        runner.sample(num_samples=args.num_samples)


if __name__ == '__main__':
    main()
