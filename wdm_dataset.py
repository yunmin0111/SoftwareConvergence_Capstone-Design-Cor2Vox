"""
WDM Dataset for Cor2Vox.
Replaces resize with 3D Discrete Wavelet Transform.

Original: 256³ → resize → 128³ (lossy)
WDM:      256³ → DWT → 128³ x 8ch (lossless)

Usage: Place in Cor2Vox/datasets/ folder.
       Import and register in main.py.
"""

import os
import torch
import numpy as np
import nibabel as nib
import pywt
from torch.utils.data import Dataset
from Register import Registers


def dwt3d(volume, wavelet='haar'):
    """3D DWT: (H,W,D) → (8, H/2, W/2, D/2)"""
    coeffs = pywt.dwtn(volume, wavelet)
    keys = sorted(coeffs.keys())
    return np.stack([coeffs[k] for k in keys], axis=0).astype(np.float32)


def iwt3d(subbands, wavelet='haar'):
    """3D IWT: (8, H, W, D) → (H*2, W*2, D*2)"""
    keys = sorted(['aaa', 'aad', 'ada', 'add', 'daa', 'dad', 'dda', 'ddd'])
    coeffs = {k: subbands[i] for i, k in enumerate(keys)}
    return pywt.idwtn(coeffs, wavelet).astype(np.float32)


def pair_file(base_files, target_folder):
    """Match files between folders by subject ID."""
    import re
    target_files = sorted([
        os.path.join(target_folder, f)
        for f in os.listdir(target_folder) if f.endswith('.nii.gz')
    ])
    
    base_ids = {}
    for f in base_files:
        match = re.search(r'CC(\d+)', os.path.basename(f))
        if match:
            base_ids[match.group(1)] = f
    
    target_ids = {}
    for f in target_files:
        match = re.search(r'CC(\d+)', os.path.basename(f))
        if match:
            target_ids[match.group(1)] = f
    
    paired = []
    for f in base_files:
        match = re.search(r'CC(\d+)', os.path.basename(f))
        if match and match.group(1) in target_ids:
            paired.append(target_ids[match.group(1)])
        else:
            paired.append(None)
    
    return paired


@Registers.datasets.register_with_name('wdm_c2v')
class WdmC2vDataset(Dataset):
    """
    WDM version of C2vDataset.
    
    Instead of resizing 256³ → 128³, applies DWT to get 128³ × 8ch.
    
    Channel layout:
      x (MRI target):     8ch wavelet (all subbands)
      y (Sc shape prior): 8ch wavelet (all subbands)  
      condition:          4ch (LLL only for Sp, Sw, E, R)
    
    UNet input = y(8) + condition(4) + x_t(8) = 20ch
    UNet output = 8ch (wavelet MRI)
    """
    
    def __init__(self, dataset_config, stage='train'):
        self.wavelet = 'haar'
        
        if stage == 'train':
            suffix = 'Tr'
        elif stage == 'val':
            suffix = 'Val'
        else:
            suffix = 'Ts'
        
        # Get folder paths
        img_folder = dataset_config['img_folder'] + suffix
        shape_folder = dataset_config['shape_folder'] + suffix
        shape_folder_2 = dataset_config['shape_folder_2'] + suffix
        
        cond_folders = []
        for key in ['condition_folder', 'condition_folder_2',
                     'condition_folder_3', 'condition_folder_4']:
            if key in dataset_config:
                cond_folders.append(dataset_config[key] + suffix)
        
        # Load file lists
        self.img_files = sorted([
            os.path.join(img_folder, f)
            for f in os.listdir(img_folder) if f.endswith('.nii.gz')
        ])
        
        self.shape_files = pair_file(self.img_files, shape_folder)
        self.shape_files_2 = pair_file(self.img_files, shape_folder_2)
        
        self.cond_files = []
        for cf in cond_folders:
            self.cond_files.append(pair_file(self.img_files, cf))
        
        print(f"WdmC2vDataset [{stage}]: {len(self.img_files)} samples, "
              f"{len(cond_folders)} conditions, wavelet={self.wavelet}")
    
    def __len__(self):
        return len(self.img_files)
    
    def _load(self, path):
        return nib.load(path).get_fdata().astype(np.float32)
    
    def _create_cortex_sdf(self, sp, sw):
        """Merge pial and white SDF into cortex SDF."""
        abs_sp = np.abs(sp)
        abs_sw = np.abs(sw)
        sc = np.where(abs_sp < abs_sw, sp, sw)
        return sc
    
    def __getitem__(self, idx):
        # Load 256³ volumes (NO resize!)
        mri = self._load(self.img_files[idx])
        sp = self._load(self.shape_files[idx])
        sw = self._load(self.shape_files_2[idx])
        
        # Create cortex SDF
        sc = self._create_cortex_sdf(sp, sw)
        
        # === Apply DWT ===
        
        # MRI target: full 8-subband wavelet (256³ → 128³ × 8ch)
        x = dwt3d(mri, self.wavelet)  # (8, 128, 128, 128)
        
        # Shape prior Sc: full 8-subband wavelet
        y = dwt3d(sc, self.wavelet)   # (8, 128, 128, 128)
        
        # Conditions: LLL only (keeps channels manageable)
        cond_list = []
        for cf_list in self.cond_files:
            vol = self._load(cf_list[idx])
            coeffs = pywt.dwtn(vol, self.wavelet)
            lll = coeffs['aaa']  # (128, 128, 128)
            cond_list.append(lll)
        
        condition = np.stack(cond_list, axis=0)  # (4, 128, 128, 128)
        
        # Convert to tensors
        return {
            'x': torch.from_numpy(x),           # (8, 128, 128, 128)
            'y': torch.from_numpy(y),           # (8, 128, 128, 128)
            'condition': torch.from_numpy(condition),  # (4, 128, 128, 128)
        }
