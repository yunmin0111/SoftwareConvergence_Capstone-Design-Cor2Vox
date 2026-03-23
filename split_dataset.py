"""
Split preprocessed data into Train/Val/Test sets for Cor2Vox training.

Creates the folder structure required by Cor2Vox with Tr/Val/Ts suffixes.
Also prepares MRI files by converting .nii to .nii.gz and matching filenames.

Input:
    - Preprocessed SDF, edge map, ribbon mask volumes
    - Original MRI .nii files
Output:
    - Tr/Val/Ts split folders with symlinks

Usage:
    python split_dataset.py \
        --data_dir /data/yunmin0111/dataset \
        --mri_dir /data/datasets/CamCAN/mri \
        --train_ratio 0.8 \
        --val_ratio 0.1 \
        --test_ratio 0.1
"""

import os
import re
import argparse
import numpy as np
import nibabel as nib
from tqdm import tqdm
import logging
import shutil

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
LOG = logging.getLogger(__name__)


def extract_subject_id(filename):
    """
    Extract numeric subject ID from filename.
    e.g., 'sub-CC110037_defaced_T1.nii.gz' -> '110037'
          'wmsub-CC110037_defaced_T1.nii' -> '110037'

    Args:
        filename: string filename

    Returns:
        numeric_id: string of digits
    """
    digits = re.findall(r'\d+', filename)
    return ''.join(digits)


def main():
    parser = argparse.ArgumentParser(
        description="Split preprocessed data into Train/Val/Test for Cor2Vox"
    )
    parser.add_argument(
        '--data_dir', type=str, required=True,
        help='Directory containing preprocessed data '
             '(sdf_pial/, sdf_white/, edge_map/, ribbon_mask/)'
    )
    parser.add_argument(
        '--mri_dir', type=str, required=True,
        help='Directory containing original MRI .nii files'
    )
    parser.add_argument(
        '--train_ratio', type=float, default=0.8,
        help='Training set ratio (default: 0.8)'
    )
    parser.add_argument(
        '--val_ratio', type=float, default=0.1,
        help='Validation set ratio (default: 0.1)'
    )
    parser.add_argument(
        '--test_ratio', type=float, default=0.1,
        help='Test set ratio (default: 0.1)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    args = parser.parse_args()

    assert abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) < 1e-6, \
        "Train/Val/Test ratios must sum to 1.0"

    # --- Step 1: Prepare MRI files ---
    mri_output_dir = os.path.join(args.data_dir, 'mri')
    os.makedirs(mri_output_dir, exist_ok=True)

    LOG.info("Step 1: Preparing MRI files...")
    mri_files = sorted([
        f for f in os.listdir(args.mri_dir)
        if f.endswith('.nii') or f.endswith('.nii.gz')
    ])

    # Convert MRI files to .nii.gz with matching subject ID filename
    mri_subject_map = {}  # subject_id -> mri filename
    for mri_file in tqdm(mri_files, desc="Preparing MRI"):
        subject_id = extract_subject_id(mri_file)
        src_path = os.path.join(args.mri_dir, mri_file)
        dst_filename = f"sub-CC{subject_id}_defaced_T1.nii.gz"
        dst_path = os.path.join(mri_output_dir, dst_filename)

        if not os.path.exists(dst_path):
            try:
                img = nib.load(src_path)
                nib.save(img, dst_path)
                mri_subject_map[subject_id] = dst_filename
            except Exception as e:
                LOG.error(f"[ERROR] MRI {mri_file}: {e}")
        else:
            mri_subject_map[subject_id] = dst_filename

    LOG.info(f"Prepared {len(mri_subject_map)} MRI files")

    # --- Step 2: Find common subjects across all modalities ---
    data_folders = {
        'sdf_pial': os.path.join(args.data_dir, 'sdf_pial'),
        'sdf_white': os.path.join(args.data_dir, 'sdf_white'),
        'edge_map': os.path.join(args.data_dir, 'edge_map'),
        'ribbon_mask': os.path.join(args.data_dir, 'ribbon_mask'),
        'mri': mri_output_dir,
    }

    # Get subject IDs from each modality
    subject_sets = {}
    for name, folder in data_folders.items():
        if os.path.exists(folder):
            files = [f for f in os.listdir(folder) if f.endswith('.nii.gz')]
            ids = set(extract_subject_id(f) for f in files)
            subject_sets[name] = ids
            LOG.info(f"  {name}: {len(ids)} subjects")
        else:
            LOG.warning(f"  {name}: folder not found ({folder})")
            subject_sets[name] = set()

    # Find intersection of all modalities
    common_subjects = set.intersection(*subject_sets.values())
    common_subjects = sorted(list(common_subjects))
    LOG.info(f"Common subjects across all modalities: {len(common_subjects)}")

    if len(common_subjects) == 0:
        LOG.error("No common subjects found! Check your data.")
        return

    # --- Step 3: Split into Train/Val/Test ---
    np.random.seed(args.seed)
    indices = np.random.permutation(len(common_subjects))

    n_train = int(len(common_subjects) * args.train_ratio)
    n_val = int(len(common_subjects) * args.val_ratio)

    train_subjects = [common_subjects[i] for i in indices[:n_train]]
    val_subjects = [common_subjects[i] for i in indices[n_train:n_train + n_val]]
    test_subjects = [common_subjects[i] for i in indices[n_train + n_val:]]

    LOG.info(f"Split: Train={len(train_subjects)}, Val={len(val_subjects)}, Test={len(test_subjects)}")

    # Save split lists
    for split_name, split_subjects in [
        ('train', train_subjects),
        ('val', val_subjects),
        ('test', test_subjects)
    ]:
        split_path = os.path.join(args.data_dir, f'{split_name}_subjects.txt')
        with open(split_path, 'w') as f:
            for s in sorted(split_subjects):
                f.write(s + '\n')
        LOG.info(f"Saved {split_name} subject list to {split_path}")

    # --- Step 4: Create Tr/Val/Ts folders ---
    split_suffix = {
        'train': 'Tr',
        'val': 'Val',
        'test': 'Ts',
    }

    split_map = {
        'train': train_subjects,
        'val': val_subjects,
        'test': test_subjects,
    }

    # For each data folder, create Tr/Val/Ts subfolders with symlinks
    for folder_name, folder_path in data_folders.items():
        if not os.path.exists(folder_path):
            continue

        # Get all files in this folder
        all_files = sorted([
            f for f in os.listdir(folder_path)
            if f.endswith('.nii.gz')
        ])

        # Build a mapping: subject_id -> filename
        id_to_file = {}
        for f in all_files:
            sid = extract_subject_id(f)
            id_to_file[sid] = f

        for split_name, subjects in split_map.items():
            suffix = split_suffix[split_name]
            split_dir = os.path.join(args.data_dir, f"{folder_name}{suffix}")
            os.makedirs(split_dir, exist_ok=True)

            count = 0
            for sid in subjects:
                if sid in id_to_file:
                    src = os.path.join(folder_path, id_to_file[sid])
                    dst = os.path.join(split_dir, id_to_file[sid])

                    # Create symlink (avoid duplicating large files)
                    if not os.path.exists(dst):
                        os.symlink(os.path.abspath(src), dst)
                    count += 1

            LOG.info(f"  {folder_name}{suffix}: {count} files")

    # --- Step 5: Print final config for c2v.yaml ---
    LOG.info(f"\n{'='*60}")
    LOG.info("Update your configs/c2v.yaml with these paths:")
    LOG.info(f"{'='*60}")
    LOG.info(f"""
data:
  dataset_config:
    img_folder: '{args.data_dir}/mri'
    shape_folder: '{args.data_dir}/sdf_pial'
    shape_folder_2: '{args.data_dir}/sdf_white'
    condition_folder: '{args.data_dir}/sdf_pial'
    condition_folder_2: '{args.data_dir}/sdf_white'
    condition_folder_3: '{args.data_dir}/edge_map'
    condition_folder_4: '{args.data_dir}/ribbon_mask'
    """)
    LOG.info(f"{'='*60}")
    LOG.info("Preprocessing complete!")


if __name__ == '__main__':
    main()
