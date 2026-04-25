"""
Wavelet-based Dataset for WDM-Cor2Vox.

Wraps the original C2vDataset and applies 3D Discrete Wavelet Transform
to all data, converting 256³ x 1ch → 128³ x 8ch.

This preserves ALL information from the original 256³ resolution
while keeping the spatial dimensions at 128³ for memory efficiency.
"""

import torch
import numpy as np
import pywt
from torch.utils.data import Dataset


def dwt3d(volume, wavelet='haar'):
    """
    Apply 3D Discrete Wavelet Transform.
    
    Input:  (H, W, D) numpy array, e.g. (256, 256, 256)
    Output: (8, H/2, W/2, D/2) numpy array, e.g. (8, 128, 128, 128)
    
    The 8 channels are the wavelet subbands:
      aaa (LLL): low-freq in all 3 axes = overall shape
      aad (LLH): high-freq in z = z-axis details
      ada (LHL): high-freq in y = y-axis details
      add (LHH): high-freq in y,z
      daa (HLL): high-freq in x = x-axis details
      dad (HLH): high-freq in x,z
      dda (HHL): high-freq in x,y
      ddd (HHH): high-freq in all 3 = finest details
    """
    coeffs = pywt.dwtn(volume, wavelet)
    # Sort keys for consistent ordering
    keys = sorted(coeffs.keys())
    subbands = np.stack([coeffs[k] for k in keys], axis=0)
    return subbands.astype(np.float32)


def iwt3d(subbands, wavelet='haar'):
    """
    Apply 3D Inverse Discrete Wavelet Transform.
    
    Input:  (8, H, W, D) numpy array, e.g. (8, 128, 128, 128)
    Output: (H*2, W*2, D*2) numpy array, e.g. (256, 256, 256)
    """
    keys = sorted(['aaa', 'aad', 'ada', 'add', 'daa', 'dad', 'dda', 'ddd'])
    coeffs = {k: subbands[i] for i, k in enumerate(keys)}
    volume = pywt.idwtn(coeffs, wavelet)
    return volume.astype(np.float32)


def dwt3d_tensor(tensor, wavelet='haar'):
    """
    Apply DWT to a PyTorch tensor.
    
    Input:  (C, H, W, D) tensor
    Output: (C*8, H/2, W/2, D/2) tensor
    
    Each channel is independently transformed.
    """
    C = tensor.shape[0]
    result = []
    for c in range(C):
        vol = tensor[c].numpy()
        subbands = dwt3d(vol, wavelet)  # (8, H/2, W/2, D/2)
        result.append(subbands)
    # Stack: (C, 8, H/2, W/2, D/2) → reshape to (C*8, H/2, W/2, D/2)
    result = np.concatenate(result, axis=0)
    return torch.from_numpy(result)


def iwt3d_tensor(tensor, num_original_channels, wavelet='haar'):
    """
    Apply IWT to a PyTorch tensor.
    
    Input:  (C*8, H, W, D) tensor
    Output: (C, H*2, W*2, D*2) tensor
    """
    total_ch = tensor.shape[0]
    C = num_original_channels
    assert total_ch == C * 8, f"Expected {C*8} channels, got {total_ch}"
    
    result = []
    for c in range(C):
        subbands = tensor[c*8:(c+1)*8].numpy()
        vol = iwt3d(subbands, wavelet)
        result.append(vol)
    return torch.from_numpy(np.stack(result, axis=0))


class WaveletC2vDataset(Dataset):
    """
    Wrapper around C2vDataset that applies DWT to all data.
    
    Original: loads 256³ data, resizes to 128³ (lossy!)
    This:     loads 256³ data, applies DWT to get 128³ x 8ch (lossless!)
    
    For shape/condition inputs (SDF, Edge, Ribbon):
      - Uses only LLL subband (1ch per input) to keep channels manageable
      - LLL contains the overall structure which is most important for conditioning
    
    For MRI target (x_0) and shape prior (Sc):
      - Uses all 8 subbands to preserve full information
      - The model learns to generate all wavelet coefficients
    """
    
    def __init__(self, config, stage='train'):
        """
        Args:
            config: dataset config dict
            stage: 'train', 'val', or 'test'
        """
        import os
        import nibabel as nib
        from datasets.dataset_utils import pair_file
        
        self.config = config
        self.stage = stage
        self.wavelet = 'haar'
        
        # Determine folder suffixes
        if stage == 'train':
            suffix = 'Tr'
        elif stage == 'val':
            suffix = 'Val'
        else:
            suffix = 'Ts'
        
        # Get file paths
        img_folder = config['img_folder'] + suffix
        shape_folder = config['shape_folder'] + suffix
        shape_folder_2 = config['shape_folder_2'] + suffix
        
        condition_folders = []
        for key in ['condition_folder', 'condition_folder_2', 
                     'condition_folder_3', 'condition_folder_4']:
            if key in config:
                condition_folders.append(config[key] + suffix)
        
        # Pair files
        self.img_files = sorted([os.path.join(img_folder, f) 
                                  for f in os.listdir(img_folder) if f.endswith('.nii.gz')])
        self.shape_files = pair_file(self.img_files, shape_folder)
        self.shape_files_2 = pair_file(self.img_files, shape_folder_2)
        
        self.condition_files = []
        for cf in condition_folders:
            self.condition_files.append(pair_file(self.img_files, cf))
        
        print(f"WaveletC2vDataset [{stage}]: {len(self.img_files)} samples, "
              f"{len(condition_folders)} conditions")
    
    def __len__(self):
        return len(self.img_files)
    
    def _load_volume(self, path):
        """Load a NIfTI volume as numpy array."""
        import nibabel as nib
        return nib.load(path).get_fdata().astype(np.float32)
    
    def __getitem__(self, idx):
        # Load all volumes at original 256³ resolution
        mri = self._load_volume(self.img_files[idx])
        sp = self._load_volume(self.shape_files[idx])
        sw = self._load_volume(self.shape_files_2[idx])
        
        conditions = []
        for cf_list in self.condition_files:
            conditions.append(self._load_volume(cf_list[idx]))
        
        # Create cortex SDF (Sc) from Sp and Sw
        # Same logic as original create_cortex_sdf
        abs_sp = np.abs(sp)
        abs_sw = np.abs(sw)
        sc = np.where(abs_sp < abs_sw, sp, sw)
        
        # Apply DWT to MRI (target) - full 8 subbands
        mri_wavelet = dwt3d(mri, self.wavelet)  # (8, 128, 128, 128)
        
        # Apply DWT to Sc (shape prior) - full 8 subbands for each channel
        sc_wavelet = dwt3d(sc, self.wavelet)  # (8, 128, 128, 128)
        
        # Apply DWT to conditions - LLL only for each
        cond_list = []
        for cond in conditions:
            cond_coeffs = pywt.dwtn(cond, self.wavelet)
            lll = cond_coeffs['aaa']  # Only LLL subband
            cond_list.append(lll)
        
        # Also add Sp, Sw LLL as shape context
        sp_coeffs = pywt.dwtn(sp, self.wavelet)
        sw_coeffs = pywt.dwtn(sw, self.wavelet)
        shape_context = np.stack([sp_coeffs['aaa'], sw_coeffs['aaa']], axis=0)  # (2, 128, 128, 128)
        
        # Stack conditions: Sp_LLL, Sw_LLL, E_LLL, R_LLL
        condition_stack = np.stack(cond_list, axis=0)  # (4, 128, 128, 128)
        
        # Convert to tensors
        x = torch.from_numpy(mri_wavelet)          # (8, 128, 128, 128) - target
        y = torch.from_numpy(sc_wavelet)            # (8, 128, 128, 128) - shape prior
        context = torch.from_numpy(shape_context)   # (2, 128, 128, 128) - Sp, Sw LLL
        condition = torch.from_numpy(condition_stack) # (4, 128, 128, 128) - conditions LLL
        
        return {
            'x': x,           # MRI in wavelet space (8ch)
            'y': y,           # Sc in wavelet space (8ch)  
            'context': context,    # Shape context LLL (2ch)
            'condition': condition, # Conditions LLL (4ch)
        }


# Test code
if __name__ == '__main__':
    import nibabel as nib
    
    # Test DWT/IWT roundtrip
    print("Testing DWT/IWT roundtrip...")
    vol = np.random.randn(256, 256, 256).astype(np.float32)
    
    subbands = dwt3d(vol)
    print(f"Original: {vol.shape}")
    print(f"DWT: {subbands.shape}")
    print(f"  Total elements: orig={vol.size}, dwt={subbands.size}")
    
    reconstructed = iwt3d(subbands)
    print(f"Reconstructed: {reconstructed.shape}")
    
    error = np.abs(vol - reconstructed).max()
    print(f"Max reconstruction error: {error}")
    print(f"Lossless: {error < 1e-5}")
    
    # Test with real data
    import os
    test_dir = '/data/yunmin0111/dataset_v3/mriTr/'
    if os.path.exists(test_dir):
        f = sorted(os.listdir(test_dir))[0]
        data = nib.load(os.path.join(test_dir, f)).get_fdata().astype(np.float32)
        print(f"\nReal MRI: {f}")
        print(f"  Shape: {data.shape}, Range: [{data.min():.4f}, {data.max():.4f}]")
        
        wb = dwt3d(data)
        print(f"  DWT shape: {wb.shape}")
        for i, k in enumerate(sorted(['aaa','aad','ada','add','daa','dad','dda','ddd'])):
            print(f"    {k}: range=[{wb[i].min():.4f}, {wb[i].max():.4f}], "
                  f"energy={np.sum(wb[i]**2):.2f}")
        
        rec = iwt3d(wb)
        err = np.abs(data - rec).max()
        print(f"  Reconstruction error: {err}")
