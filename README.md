# Cor2Vox Preprocessing Pipeline

CamCAN FreeSurfer 데이터를 Cor2Vox 학습에 필요한 형태로 전처리하는 파이프라인입니다.

## Overview

[Cor2Vox](https://github.com/ai-med/Cor2Vox) 모델은 다음 6가지 입력 데이터가 필요합니다:

| 데이터 | config 키 | 설명 |
|--------|-----------|------|
| MRI 원본 | `img_folder` | 타겟 이미지 (Ground Truth) |
| SDF Pial | `shape_folder` | Pial surface의 Signed Distance Function |
| SDF White | `shape_folder_2` | White surface의 SDF |
| SDF Pial (condition) | `condition_folder` | 보조 조건 1 |
| SDF White (condition) | `condition_folder_2` | 보조 조건 2 |
| Edge Map | `condition_folder_3` | Surface mesh edge 볼륨 |
| Ribbon Mask | `condition_folder_4` | Cortical ribbon 이진 마스크 |

## Prerequisites

```bash
pip install -r requirements.txt
```

## Data Structure

### Input (FreeSurfer)
```
/data/datasets/CamCAN/
├── freesurfer/
│   ├── sub-CC110033_defaced_T1/
│   │   ├── surf/  (lh.pial, rh.pial, lh.white, rh.white)
│   │   └── mri/   (T1.mgz)
│   └── ...
└── mri/
    ├── wmsub-CC110033_defaced_T1.nii
    └── ...
```

### Output (Cor2Vox ready)
```
/data/yunmin0111/dataset/
├── sdf_pial/        (SDF from pial surfaces)
├── sdf_white/       (SDF from white surfaces)
├── edge_map/        (Binary edge map)
├── ribbon_mask/     (Cortical ribbon mask)
├── mri/             (Original MRI .nii.gz)
├── sdf_pialTr/      (Train split symlinks)
├── sdf_pialVal/     (Val split symlinks)
├── sdf_pialTs/      (Test split symlinks)
└── ...
```

## Usage

### Step 1: Generate SDF volumes
```bash
python generate_sdf.py \
    --fs_dir /data/datasets/CamCAN/freesurfer \
    --output_dir /data/yunmin0111/dataset \
    --num_workers 4
```

### Step 2: Generate Edge Map
```bash
python generate_edge_map.py \
    --fs_dir /data/datasets/CamCAN/freesurfer \
    --output_dir /data/yunmin0111/dataset \
    --num_workers 4
```

### Step 3: Generate Cortical Ribbon Mask
```bash
python generate_ribbon_mask.py \
    --sdf_pial_dir /data/yunmin0111/dataset/sdf_pial \
    --sdf_white_dir /data/yunmin0111/dataset/sdf_white \
    --output_dir /data/yunmin0111/dataset/ribbon_mask
```

### Step 4: Prepare MRI and Train/Val/Test split
```bash
python split_dataset.py \
    --data_dir /data/yunmin0111/dataset \
    --mri_dir /data/datasets/CamCAN/mri \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1
```

## Running on SERAPH (KHU GPU Cluster)

```bash
# Clone this repo
cd /data/yunmin0111
git clone https://github.com/yunmin0111/SoftwareConvergence_Capstone-Design-Cor2Vox.git
cd SoftwareConvergence_Capstone-Design-Cor2Vox

# Submit preprocessing job
sbatch run_preprocess.sh
```

## Reference

- Paper: [3D Shape-to-Image Brownian Bridge Diffusion for Brain MRI Synthesis from Cortical Surfaces](https://arxiv.org/abs/2502.12742)
- Code: [ai-med/Cor2Vox](https://github.com/ai-med/Cor2Vox)
- Data: [CamCAN Dataset](https://www.cam-can.org/)
