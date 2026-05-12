"""
Latent Diffusion Model (LDM) Runner for Cor2Vox.

Two-phase training:
  Phase 1: Train 3D Autoencoder (128³ MRI → 32³ latent → 128³ MRI)
  Phase 2: Train Brownian Bridge Diffusion in latent space

The diffusion operates on 32³ × 4ch latent instead of 128³ × 1ch pixel space.
This drastically reduces memory and computation.

Usage:
  Phase 1: python ldm_runner.py --config configs/c2v_ldm.yaml --gpu_ids 0 --train_ae
  Phase 2: python ldm_runner.py --config configs/c2v_ldm.yaml --gpu_ids 0 --train
  Sample:  python ldm_runner.py --config configs/c2v_ldm.yaml --gpu_ids 0 --resume_model path.pth
"""

import os, sys, yaml, torch, numpy as np, nibabel as nib, argparse
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from autoencoder3d import AutoEncoder3D
from light_unet import LightUNet3D


class LatentBBModel(torch.nn.Module):
    """Brownian Bridge operating in latent space (32³ × 4ch)."""
    
    def __init__(self, params, unet_config):
        super().__init__()
        self.num_timesteps = params['num_timesteps']
        self.loss_type = params['loss_type']
        self.eta = params['eta']
        self.max_var = params['max_var']
        self.skip_sample = params['skip_sample']
        self.sample_type = params['sample_type']
        self.sample_step = params['sample_step']
        self.mt_type = params['mt_type']
        self.register_schedule()
        self.denoise_fn = LightUNet3D(**unet_config)
    
    def register_schedule(self):
        T = self.num_timesteps
        m_t = np.linspace(0.001, 0.999, T + 1) if self.mt_type == 'linear' else np.sin(np.linspace(0, np.pi / 2, T + 1))
        m_tminus = np.append(0, m_t[:-1])
        var_t = 2. * (m_t - m_t ** 2) * self.max_var
        var_tminus = np.append(0., var_t[1:] * ((m_t[:-1] - m_tminus[:-1]) / (m_t[1:] - m_tminus[:-1] + 1e-8)) ** 2)
        to_t = lambda x: torch.tensor(x, dtype=torch.float32)
        self.register_buffer('m_t', to_t(m_t))
        self.register_buffer('variance_t', to_t(var_t))
        self.register_buffer('variance_tminus', to_t(var_tminus))
        self.register_buffer('posterior_variance_t', to_t(var_t - var_tminus))
    
    def forward(self, x, y, condition=None):
        b = x.shape[0]
        t = torch.randint(1, self.num_timesteps + 1, (b,), device=x.device).long()
        return self.p_losses(x, y, condition=condition, t=t)
    
    def p_losses(self, x_0, y, condition=None, t=None):
        b = x_0.shape[0]
        m_t = self.m_t[t].view(b, 1, 1, 1, 1)
        var_t = self.variance_t[t].view(b, 1, 1, 1, 1)
        noise = torch.randn_like(x_0)
        x_t = (1. - m_t) * x_0 + m_t * y + torch.sqrt(var_t) * noise
        ctx = torch.cat((y, condition), dim=1) if condition is not None else y
        inp = torch.cat((x_t, ctx), dim=1)
        obj_recon = self.denoise_fn(inp, timesteps=t)
        obj = m_t * (y - x_0) + torch.sqrt(var_t) * noise
        return (obj - obj_recon).abs().mean() if self.loss_type == 'l1' else ((obj - obj_recon) ** 2).mean()
    
    @torch.no_grad()
    def p_sample_loop(self, y, condition=None):
        device, b = y.device, y.shape[0]
        x_t = y.clone()
        steps = list(reversed(list(np.linspace(0, self.num_timesteps, self.sample_step + 1, dtype=int)))) if self.skip_sample and self.sample_type == 'linear' else list(reversed(range(self.num_timesteps + 1)))
        for i in tqdm(range(len(steps) - 1), desc="sampling"):
            cs, ns = steps[i], steps[i + 1]
            t = torch.full((b,), cs, device=device, dtype=torch.long)
            ctx = torch.cat((y, condition), dim=1) if condition is not None else y
            inp = torch.cat((x_t, ctx), dim=1)
            obj_recon = self.denoise_fn(inp, timesteps=t)
            x0r = torch.clamp(x_t - obj_recon, -1., 1.)
            if ns == 0:
                x_t = x0r
            else:
                m_t, m_nt = self.m_t[cs], self.m_t[ns]
                vt, vnt = self.variance_t[cs], self.variance_t[ns]
                s2 = torch.clamp((vt - vnt * (1. - m_t) ** 2 / ((1. - m_nt) ** 2 + 1e-8)) * vnt / (vt + 1e-8), min=0)
                st = torch.sqrt(s2) * self.eta
                x_t = (1. - m_nt) * x0r + m_nt * y + torch.sqrt(torch.clamp((vnt - s2) / (vt + 1e-8), min=0)) * (x_t - (1. - m_t) * x0r - m_t * y) + st * torch.randn_like(x_t)
        return x_t


class LDMRunner:
    def __init__(self, config, gpu_id=0):
        self.config = config
        self.device = torch.device(f'cuda:{gpu_id}')
        
        mn = config['model']['model_name']
        self.ckpt_dir = f'results/c2v/{mn}/checkpoint'
        self.sample_dir = f'results/c2v/{mn}/samples'
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)
        
        # Autoencoder
        ae_config = config['model'].get('autoencoder', {})
        self.ae = AutoEncoder3D(
            in_channels=1,
            latent_dim=ae_config.get('latent_dim', 4),
            base_channels=ae_config.get('base_channels', 32),
            kl_weight=ae_config.get('kl_weight', 1e-6)
        ).to(self.device)
        
        ae_total = sum(p.numel() for p in self.ae.parameters())
        print(f"AutoEncoder parameters: {ae_total / 1e6:.2f}M")
        
        # Diffusion model (operates in latent space)
        bb = config['model']['BB']
        self.model = LatentBBModel(bb['params'], bb['UNetParams']).to(self.device)
        dm_total = sum(p.numel() for p in self.model.parameters())
        print(f"Diffusion model parameters: {dm_total / 1e6:.2f}M")
        print(f"Total parameters: {(ae_total + dm_total) / 1e6:.2f}M")
        
        # Optimizers
        self.ae_optimizer = Adam(self.ae.parameters(), lr=bb['optimizer']['lr'])
        self.dm_optimizer = Adam(self.model.parameters(), lr=bb['optimizer']['lr'])
        self.ae_scheduler = ReduceLROnPlateau(self.ae_optimizer, factor=0.5, patience=20)
        self.dm_scheduler = ReduceLROnPlateau(self.dm_optimizer, factor=0.5, patience=20)
        
        self.start_epoch = 0
    
    def _load_ae(self, path):
        state = torch.load(path, map_location='cpu')
        if 'ae' in state:
            self.ae.load_state_dict(state['ae'])
            print(f"Loaded AE from {path}")
        else:
            self.ae.load_state_dict(state, strict=False)
    
    def _load_dm(self, path):
        state = torch.load(path, map_location='cpu')
        if 'model' in state:
            self.model.load_state_dict(state['model'])
            if 'epoch' in state:
                self.start_epoch = state['epoch']
            print(f"Loaded DM from {path}, epoch {self.start_epoch}")
        else:
            self.model.load_state_dict(state, strict=False)
    
    def _loader(self, stage):
        from datasets.dataset import C2vDataset
        class DictToObj:
            def __init__(self, d):
                for k, v in d.items(): setattr(self, k, v)
        ds = C2vDataset(DictToObj(self.config['data']['dataset_config']), stage=stage)
        return DataLoader(ds, batch_size=1, shuffle=(stage == 'train'),
                         num_workers=4, pin_memory=True, drop_last=(stage == 'train'))
    
    def train_ae(self, max_epochs=100):
        """Phase 1: Train autoencoder."""
        loader = self._loader('train')
        print(f"=== Phase 1: Training AutoEncoder for {max_epochs} epochs ===")
        
        for epoch in range(max_epochs):
            self.ae.train()
            epoch_loss = 0
            pbar = tqdm(loader, desc=f"AE Epoch [{epoch + 1}/{max_epochs}]")
            
            for batch in pbar:
                x = batch[0][0].to(self.device)  # MRI (B, 1, 128, 128, 128)
                
                self.ae_optimizer.zero_grad()
                x_recon, mean, logvar = self.ae(x)
                loss, recon_loss, kl_loss = self.ae.compute_loss(x, x_recon, mean, logvar)
                loss.backward()
                self.ae_optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}", recon=f"{recon_loss.item():.4f}")
            
            avg = epoch_loss / len(loader)
            print(f"AE Epoch {epoch + 1}: avg_loss={avg:.4f}")
            
            if (epoch + 1) % 5 == 0:
                val_loss = self._validate_ae()
                self.ae_scheduler.step(val_loss)
                print(f"  val_loss={val_loss:.4f}")
            
            state = {'ae': self.ae.state_dict(), 'epoch': epoch + 1}
            torch.save(state, os.path.join(self.ckpt_dir, 'ae_last.pth'))
            if (epoch + 1) % 20 == 0:
                torch.save(state, os.path.join(self.ckpt_dir, f'ae_epoch_{epoch + 1}.pth'))
    
    def _validate_ae(self):
        loader = self._loader('val')
        self.ae.eval()
        total = 0
        with torch.no_grad():
            for batch in loader:
                x = batch[0][0].to(self.device)
                x_recon, mean, logvar = self.ae(x)
                loss, _, _ = self.ae.compute_loss(x, x_recon, mean, logvar)
                total += loss.item()
        return total / len(loader)
    
    def train_dm(self, max_epochs=400):
        """Phase 2: Train diffusion in latent space (AE frozen)."""
        # Load pretrained AE
        ae_path = os.path.join(self.ckpt_dir, 'ae_last.pth')
        if os.path.exists(ae_path):
            self._load_ae(ae_path)
        else:
            print("ERROR: Train autoencoder first! Use --train_ae")
            return
        
        self.ae.eval()
        for p in self.ae.parameters():
            p.requires_grad = False
        
        loader = self._loader('train')
        print(f"=== Phase 2: Training Diffusion in Latent Space ===")
        print(f"Starting from epoch {self.start_epoch}")
        
        for epoch in range(self.start_epoch, max_epochs):
            self.model.train()
            epoch_loss = 0
            pbar = tqdm(loader, desc=f"DM Epoch [{epoch + 1}/{max_epochs}]")
            
            for batch in pbar:
                x_pixel = batch[0][0].to(self.device)  # MRI
                y_pixel = batch[1][0].to(self.device)  # SDF (Sc)
                cond_pixel = batch[2][0].to(self.device)  # Ribbon mask
                
                # Encode to latent space
                with torch.no_grad():
                    z_x, _, _ = self.ae.encode(x_pixel)  # MRI latent
                    z_y, _, _ = self.ae.encode(y_pixel[:, :1])  # Sc ch0 latent
                    # For condition: just downsample ribbon mask to 32³
                    cond_latent = F.interpolate(cond_pixel, size=z_x.shape[2:], mode='trilinear', align_corners=False)
                
                self.dm_optimizer.zero_grad()
                loss = self.model(z_x, z_y, condition=cond_latent)
                loss.backward()
                self.dm_optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            
            avg = epoch_loss / len(loader)
            print(f"DM Epoch {epoch + 1}: avg_loss={avg:.4f}")
            
            if (epoch + 1) % 5 == 0:
                val_loss = self._validate_dm()
                self.dm_scheduler.step(val_loss)
                print(f"  val_loss={val_loss:.4f}")
            
            state = {'model': self.model.state_dict(), 'optimizer': self.dm_optimizer.state_dict(), 'epoch': epoch + 1}
            torch.save(state, os.path.join(self.ckpt_dir, 'dm_last.pth'))
            if (epoch + 1) % 20 == 0:
                torch.save(state, os.path.join(self.ckpt_dir, f'dm_epoch_{epoch + 1}.pth'))
    
    def _validate_dm(self):
        loader = self._loader('val')
        self.model.eval()
        total = 0
        with torch.no_grad():
            for batch in loader:
                x_pixel = batch[0][0].to(self.device)
                y_pixel = batch[1][0].to(self.device)
                cond_pixel = batch[2][0].to(self.device)
                z_x, _, _ = self.ae.encode(x_pixel)
                z_y, _, _ = self.ae.encode(y_pixel[:, :1])
                cond_latent = F.interpolate(cond_pixel, size=z_x.shape[2:], mode='trilinear', align_corners=False)
                total += self.model(z_x, z_y, condition=cond_latent).item()
        return total / len(loader)
    
    def sample(self, n=3):
        """Generate samples: diffuse in latent → decode to pixel."""
        ae_path = os.path.join(self.ckpt_dir, 'ae_last.pth')
        self._load_ae(ae_path)
        self.ae.eval()
        
        loader = self._loader('test')
        self.model.eval()
        
        for i, batch in enumerate(loader):
            if i >= n: break
            y_pixel = batch[1][0].to(self.device)
            cond_pixel = batch[2][0].to(self.device)
            
            with torch.no_grad():
                z_y, _, _ = self.ae.encode(y_pixel[:, :1])
                cond_latent = F.interpolate(cond_pixel, size=z_y.shape[2:], mode='trilinear', align_corners=False)
                
                # Diffuse in latent space
                z_syn = self.model.p_sample_loop(z_y, condition=cond_latent)
                
                # Decode back to pixel space
                syn_pixel = self.ae.decode(z_syn)
            
            syn_np = syn_pixel[0, 0].cpu().numpy()
            real_np = batch[0][0][0, 0].numpy()
            
            nib.save(nib.Nifti1Image(syn_np, np.eye(4)), f'{self.sample_dir}/sample{i}_syn.nii.gz')
            nib.save(nib.Nifti1Image(real_np, np.eye(4)), f'{self.sample_dir}/sample{i}_real.nii.gz')
            print(f"Sample {i}: syn [{syn_np.min():.4f}, {syn_np.max():.4f}], latent {z_syn.shape}")
        
        print(f"Saved to {self.sample_dir}")


def main():
    import torch.nn.functional as F
    
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--gpu_ids', default='0')
    p.add_argument('--train_ae', action='store_true', help='Phase 1: train autoencoder')
    p.add_argument('--train', action='store_true', help='Phase 2: train diffusion')
    p.add_argument('--resume_model', default='')
    p.add_argument('--num_samples', type=int, default=3)
    a = p.parse_args()
    
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    
    runner = LDMRunner(cfg, int(a.gpu_ids.split(',')[0]))
    
    if a.resume_model:
        runner._load_dm(a.resume_model)
    
    if a.train_ae:
        ae_epochs = cfg['model'].get('autoencoder', {}).get('epochs', 100)
        runner.train_ae(max_epochs=ae_epochs)
    elif a.train:
        runner.train_dm()
    else:
        runner.sample(a.num_samples)


if __name__ == '__main__':
    main()
