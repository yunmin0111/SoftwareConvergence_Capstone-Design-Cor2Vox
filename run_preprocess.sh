#!/usr/bin/bash
#SBATCH -J preprocess
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -p batch_grad
#SBATCH --account=grad
#SBATCH -t 2-0
#SBATCH -o /data/%u/logs/preprocess-%A.out

# Activate conda
source ~/.bashrc
conda activate c2v

# Set working directory
cd /data/$USER/SoftwareConvergence_Capstone-Design-Cor2Vox

# ===== Step 1: Generate SDF =====
echo "===== Step 1: Generating SDF volumes ====="
python generate_sdf.py \
    --fs_dir /data/datasets/CamCAN/freesurfer \
    --output_dir /data/$USER/dataset \
    --num_workers 8

# ===== Step 2: Generate Edge Map =====
echo "===== Step 2: Generating Edge Maps ====="
python generate_edge_map.py \
    --fs_dir /data/datasets/CamCAN/freesurfer \
    --output_dir /data/$USER/dataset \
    --num_workers 8

# ===== Step 3: Generate Ribbon Mask =====
echo "===== Step 3: Generating Ribbon Masks ====="
python generate_ribbon_mask.py \
    --sdf_pial_dir /data/$USER/dataset/sdf_pial \
    --sdf_white_dir /data/$USER/dataset/sdf_white \
    --output_dir /data/$USER/dataset/ribbon_mask

# ===== Step 4: Split Dataset =====
echo "===== Step 4: Splitting Dataset ====="
python split_dataset.py \
    --data_dir /data/$USER/dataset \
    --mri_dir /data/datasets/CamCAN/mri \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1

echo "===== All preprocessing complete! ====="
