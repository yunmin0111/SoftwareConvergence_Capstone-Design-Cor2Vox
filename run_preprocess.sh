#!/usr/bin/bash
#SBATCH -J preprocess
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -p batch_grad
#SBATCH --account=grad
#SBATCH -t 2-0
#SBATCH -o /data/%u/logs/preprocess-%A.out

# Conda 초기화 (Computing 노드에서 필수)
eval "$(/data/yunmin0111/anaconda3/bin/conda shell.bash hook)"
conda activate c2v

# 패키지 설치 (처음 한번만 실행됨)
pip install nibabel trimesh numpy scipy tqdm scikit-learn

# Set working directory
cd /data/yunmin0111/SoftwareConvergence_Capstone-Design-Cor2Vox

# ===== Step 1: Generate SDF =====
echo "===== Step 1: Generating SDF volumes ====="
python generate_sdf.py \
    --fs_dir /data/datasets/CamCAN/freesurfer \
    --output_dir /data/yunmin0111/dataset \
    --num_workers 8

# ===== Step 2: Generate Edge Map =====
echo "===== Step 2: Generating Edge Maps ====="
python generate_edge_map.py \
    --fs_dir /data/datasets/CamCAN/freesurfer \
    --output_dir /data/yunmin0111/dataset \
    --num_workers 8

# ===== Step 3: Generate Ribbon Mask =====
echo "===== Step 3: Generating Ribbon Masks ====="
python generate_ribbon_mask.py \
    --sdf_pial_dir /data/yunmin0111/dataset/sdf_pial \
    --sdf_white_dir /data/yunmin0111/dataset/sdf_white \
    --output_dir /data/yunmin0111/dataset/ribbon_mask

# ===== Step 4: Split Dataset =====
echo "===== Step 4: Splitting Dataset ====="
python split_dataset.py \
    --data_dir /data/yunmin0111/dataset \
    --mri_dir /data/datasets/CamCAN/mri \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1

echo "===== All preprocessing complete! ====="