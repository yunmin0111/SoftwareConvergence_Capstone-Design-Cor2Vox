"""
DSMamba-UNet: Lightweight 3D UNet with Selective SSM at Bottleneck

Architecture:
  - Encoder/Decoder: Depthwise Separable Conv3D (lightweight, no external packages)
  - Bottleneck: Pure-PyTorch Mamba Selective SSM (long-range dependency, no mamba-ssm needed)

Key difference from LightM-UNet:
  LightM-UNet: Mamba everywhere (95%), DW Conv minimal (5%)
  DSMamba-UNet: DS Conv dominant (85%), Mamba only at bottleneck (15%)

This is a drop-in replacement for the UNet in Cor2Vox's Brownian Bridge diffusion.
Same interface: forward(x, timesteps, context=None)

No external packages required - pure PyTorch implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange


# ============================================================
# Pure-PyTorch Selective SSM (Mamba core)
# Based on: https://github.com/johnma2006/mamba-minimal
# ============================================================

class SelectiveSSM(nn.Module):
    """
    Pure-PyTorch implementation of Mamba's Selective State Space Model.
    
    No mamba-ssm or causal-conv1d package needed!
    
    Key idea: A, B, C parameters are INPUT-DEPENDENT (selective)
    - Important voxels → B is large (absorb into memory)
    - Background voxels → B is small (ignore)
    
    Args:
        d_model: input dimension
        d_state: SSM state dimension (N in paper, default 16)
        d_conv: local convolution width (default 4)
        expand: expansion factor for inner dimension (default 2)
    """
    
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(d_model * expand)
        
        # Input projection: d_model → 2*d_inner (for x and z branches)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Local convolution (replaces causal-conv1d)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, 
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True
        )
        
        # Selective parameters (input-dependent B, C, delta)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)  # B, C, delta
        
        # A parameter (not input-dependent, learned)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))  # log for numerical stability
        
        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
    
    def forward(self, x):
        """
        Args:
            x: (B, L, D) where L=sequence length, D=d_model
        Returns:
            y: (B, L, D)
        """
        b, l, d = x.shape
        
        # Input projection: split into x and z (gate)
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_branch, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)
        
        # Local convolution
        x_branch = x_branch.transpose(1, 2)  # (B, d_inner, L)
        x_branch = self.conv1d(x_branch)[:, :, :l]  # causal: trim to original length
        x_branch = x_branch.transpose(1, 2)  # (B, L, d_inner)
        x_branch = F.silu(x_branch)
        
        # Selective parameters from input
        x_proj = self.x_proj(x_branch)  # (B, L, 2*d_state + 1)
        B = x_proj[:, :, :self.d_state]  # (B, L, N)
        C = x_proj[:, :, self.d_state:2*self.d_state]  # (B, L, N)
        delta = F.softplus(x_proj[:, :, -1])  # (B, L) - step size
        
        # A matrix (fixed, not input-dependent)
        A = -torch.exp(self.A_log)  # (d_inner, N)
        
        # Selective scan (sequential implementation)
        y = self.selective_scan(x_branch, delta, A, B, C)
        
        # Skip connection + gate
        y = y * F.silu(z)  # (B, L, d_inner)
        
        # Output projection
        return self.out_proj(y)  # (B, L, d_model)
    
    def selective_scan(self, u, delta, A, B, C):
        """
        Selective scan (sequential loop, pure PyTorch).
        
        u: (B, L, d_inner) - input
        delta: (B, L) - step size (input-dependent!)
        A: (d_inner, N) - state matrix
        B: (B, L, N) - input matrix (input-dependent!)
        C: (B, L, N) - output matrix (input-dependent!)
        
        Returns: y: (B, L, d_inner)
        """
        b, l, d_in = u.shape
        n = A.shape[1]
        
        # Discretize A and B
        # deltaA = exp(delta * A): (B, L, d_inner, N)
        deltaA = torch.exp(
            delta.unsqueeze(-1).unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
        )  # (B, L, d_inner, N)
        
        # deltaB_u = delta * B * u: (B, L, d_inner, N)
        deltaB_u = (
            delta.unsqueeze(-1).unsqueeze(-1) *  # (B, L, 1, 1)
            B.unsqueeze(2) *  # (B, L, 1, N)
            u.unsqueeze(-1)  # (B, L, d_inner, 1)
        )  # (B, L, d_inner, N)
        
        # Sequential scan
        h = torch.zeros(b, d_in, n, device=u.device, dtype=u.dtype)  # (B, d_inner, N)
        ys = []
        
        for i in range(l):
            h = deltaA[:, i] * h + deltaB_u[:, i]  # (B, d_inner, N)
            y_i = (h * C[:, i].unsqueeze(1)).sum(dim=-1)  # (B, d_inner)
            ys.append(y_i)
        
        y = torch.stack(ys, dim=1)  # (B, L, d_inner)
        
        # Add D * u (skip connection within SSM)
        y = y + u * self.D.unsqueeze(0).unsqueeze(0)
        
        return y


class MambaBlock3D(nn.Module):
    """
    Mamba block adapted for 3D volumes.
    
    Takes a 3D feature map, flattens to sequence, applies Mamba SSM,
    then reshapes back to 3D.
    
    For bottleneck at 16³: sequence length = 4096 (manageable)
    """
    
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = SelectiveSSM(d_model, d_state, d_conv, expand)
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W, D) - 3D feature map
        Returns:
            (B, C, H, W, D) - same shape
        """
        B, C, H, W, D = x.shape
        
        # Flatten spatial dims to sequence: (B, C, H*W*D) → (B, H*W*D, C)
        x_flat = x.reshape(B, C, -1).permute(0, 2, 1)  # (B, L, C) where L=H*W*D
        
        # Mamba with residual connection
        x_flat = x_flat + self.mamba(self.norm(x_flat))
        
        # Reshape back to 3D
        x_out = x_flat.permute(0, 2, 1).reshape(B, C, H, W, D)
        return x_out


# ============================================================
# Depthwise Separable Conv3D blocks (same as light_unet.py)
# ============================================================

class DSConv3D(nn.Module):
    """Depthwise Separable 3D Convolution."""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.dw = nn.Conv3d(in_ch, in_ch, kernel_size, stride=stride, 
                           padding=padding, groups=in_ch, bias=False)
        self.pw = nn.Conv3d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.GroupNorm(min(8, out_ch), out_ch)
    
    def forward(self, x):
        return self.bn(self.pw(self.dw(x)))


class DSResBlock(nn.Module):
    """Residual block with DS Conv and timestep embedding."""
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        # First conv: standard Conv3D for channel interaction
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch))
        # Second conv: DS Conv for efficiency
        self.conv2 = DSConv3D(out_ch, out_ch)
        self.act = nn.SiLU()
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.skip = nn.Conv3d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x, t):
        h = self.act(self.conv1(x))
        h = h + self.time_mlp(t)[:, :, None, None, None]
        h = self.act(self.conv2(h))
        return h + self.skip(x)


class TimeEmb(nn.Module):
    """Sinusoidal timestep embedding."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))
    
    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        return self.mlp(torch.cat([args.cos(), args.sin()], dim=-1))


# ============================================================
# DSMamba-UNet: the full model
# ============================================================

class DSMambaUNet3D(nn.Module):
    """
    DSMamba-UNet: Depthwise Separable Conv + Selective SSM at Bottleneck.
    
    Architecture:
      Encoder: DS Conv blocks (fast, lightweight)
        Level 1: 128³ × 32ch  - DS ResBlock ×2
        Level 2:  64³ × 64ch  - DS ResBlock ×2
        Level 3:  32³ × 96ch  - DS ResBlock ×2
        Down to:  16³ × 128ch
      
      Bottleneck: Mamba SSM blocks (long-range dependency)
        16³ × 128ch → flatten → 4096 sequence → Mamba ×2 → reshape
      
      Decoder: DS Conv blocks (fast, lightweight) + skip connections
        Level 3:  32³ × 96ch  - DS ResBlock ×2
        Level 2:  64³ × 64ch  - DS ResBlock ×2
        Level 1: 128³ × 32ch  - DS ResBlock ×2
    
    Drop-in replacement for UNet in Cor2Vox.
    Same interface: forward(x, timesteps, context=None)
    """
    
    def __init__(self,
                 image_size=128,
                 in_channels=8,
                 model_channels=32,
                 out_channels=1,
                 channel_mult=(1, 2, 3, 4),
                 num_res_blocks=2,
                 mamba_d_state=16,
                 mamba_d_conv=4,
                 mamba_expand=2,
                 num_mamba_blocks=2,
                 **kwargs):
        super().__init__()
        
        time_dim = model_channels * 4
        self.time_embed = TimeEmb(time_dim)
        chs = [model_channels * m for m in channel_mult]
        
        # Input projection
        self.input_conv = nn.Conv3d(in_channels, chs[0], 3, padding=1)
        
        # === Encoder (DS Conv) ===
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(len(chs) - 1):
            blks = nn.ModuleList([DSResBlock(chs[i], chs[i], time_dim) 
                                  for _ in range(num_res_blocks)])
            self.encoders.append(blks)
            self.downs.append(nn.Sequential(
                nn.Conv3d(chs[i], chs[i], 3, stride=2, padding=1),
                nn.Conv3d(chs[i], chs[i+1], 1)))
        
        # === Bottleneck (Mamba SSM) ===
        self.bottleneck_pre = DSResBlock(chs[-1], chs[-1], time_dim)
        self.mamba_blocks = nn.ModuleList([
            MambaBlock3D(chs[-1], d_state=mamba_d_state, 
                        d_conv=mamba_d_conv, expand=mamba_expand)
            for _ in range(num_mamba_blocks)
        ])
        self.bottleneck_post = DSResBlock(chs[-1], chs[-1], time_dim)
        
        # === Decoder (DS Conv) ===
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        for i in range(len(chs) - 1, 0, -1):
            self.ups.append(nn.Conv3d(chs[i], chs[i], 3, padding=1))
            blks = nn.ModuleList()
            for j in range(num_res_blocks):
                ic = chs[i] + chs[i-1] if j == 0 else chs[i-1]
                blks.append(DSResBlock(ic, chs[i-1], time_dim))
            self.decoders.append(blks)
        
        # Output projection
        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, chs[0]),
            nn.SiLU(),
            nn.Conv3d(chs[0], out_channels, 1))
    
    def forward(self, x, timesteps, context=None):
        """
        Same interface as openaimodel.py UNetModel.
        
        Args:
            x: (B, in_channels, H, W, D)
            timesteps: (B,)
            context: ignored (for compatibility)
        Returns:
            (B, out_channels, H, W, D)
        """
        t = self.time_embed(timesteps)
        h = self.input_conv(x)
        
        # Encoder
        skips = [h]
        for enc, down in zip(self.encoders, self.downs):
            for blk in enc:
                h = blk(h, t)
            skips.append(h)
            h = down(h)
        
        # Bottleneck: DS Conv + Mamba + DS Conv
        h = self.bottleneck_pre(h, t)
        for mamba in self.mamba_blocks:
            h = mamba(h)  # Mamba handles 3D→sequence→3D internally
        h = self.bottleneck_post(h, t)
        
        # Decoder
        for up, dec in zip(self.ups, self.decoders):
            h = F.interpolate(h, scale_factor=2, mode='trilinear', align_corners=False)
            h = up(h)
            sk = skips.pop()
            if h.shape != sk.shape:
                h = F.interpolate(h, size=sk.shape[2:], mode='trilinear', align_corners=False)
            h = torch.cat([h, sk], dim=1)
            for blk in dec:
                h = blk(h, t)
        
        return self.out_conv(h)


# ============================================================
# Test
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("DSMamba-UNet3D Test")
    print("=" * 60)
    
    model = DSMambaUNet3D(
        image_size=128,
        in_channels=8,
        model_channels=32,
        out_channels=1,
        channel_mult=(1, 2, 3, 4),
        num_res_blocks=2,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        num_mamba_blocks=2,
    )
    
    total = sum(p.numel() for p in model.parameters())
    mamba_params = sum(p.numel() for n, p in model.named_parameters() if 'mamba' in n)
    dsconv_params = total - mamba_params
    
    print(f"Total parameters:  {total / 1e6:.2f}M")
    print(f"  DS Conv params:  {dsconv_params / 1e6:.2f}M ({dsconv_params/total*100:.1f}%)")
    print(f"  Mamba params:    {mamba_params / 1e6:.2f}M ({mamba_params/total*100:.1f}%)")
    
    # Test forward pass
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    x = torch.randn(1, 8, 128, 128, 128).to(device)
    t = torch.tensor([500]).to(device)
    
    print(f"\nInput: {x.shape}")
    with torch.no_grad():
        y = model(x, t)
    print(f"Output: {y.shape}")
    print(f"Forward pass OK on {device}!")
    
    # Memory estimate
    if device == 'cuda':
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak GPU memory: {mem:.2f} GB")
