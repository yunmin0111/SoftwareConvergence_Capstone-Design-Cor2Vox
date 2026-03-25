"""
Generate Cortical Ribbon Mask from SDF volumes.

The cortical ribbon is the region between the pial and white matter surfaces.
This script creates a binary mask where voxels between the two surfaces are 1.

Based on Generate_cortical_ribbon_masks.py from Cor2Vox repository.

Input:
    - sdf_pial/*.nii.gz    (SDF of pial surface)
    - sdf_white/*.nii.gz   (SDF of white surface)
Output:
    - ribbon_mask/*.nii.gz (binary cortical ribbon mask)

Usage:
    python generate_ribbon_mask.py \
        --sdf_pial_dir /data/yunmin0111/dataset/sdf_pial \
        --sdf_white_dir /data/yunmin0111/dataset/sdf_white \
        --output_dir /data/yunmin0111/dataset/ribbon_mask
"""

import os
import argparse
import numpy as np
import nibabel as nib
from tqdm import tqdm
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
LOG = logging.getLogger(__name__)


def cortical_ribbon_mask_generation(sdf_pial, sdf_white):
    """
    Generate cortical ribbon mask from pial and white matter SDFs.

    The cortical ribbon is where the two SDFs have different signs
    (one positive, one negative), meaning the voxel is between the
    two surfaces.

    Args:
        sdf_pial: (H, W, D) SDF of pial surface
        sdf_white: (H, W, D) SDF of white matter surface

    Returns:
        mask: (H, W, D) binary mask (1 = cortical ribbon, 0 = outside)
    """
    # Create a mask where signs are the same
    same_sign_mask = np.sign(sdf_pial) == np.sign(sdf_white)

    sdf_combine = np.zeros_like(sdf_pial)

    # Where signs differ -> between the two surfaces -> cortical ribbon
    differing_sign_mask = ~same_sign_mask
    sdf_combine[differing_sign_mask] = 1

    # Where either SDF is exactly 0 -> on the surface -> part of ribbon
    sdf_combine[sdf_pial == 0] = 1
    sdf_combine[sdf_white == 0] = 1

    return sdf_combine


def main():
    parser = argparse.ArgumentParser(
        description="Generate Cortical Ribbon Mask from SDF volumes"
    )
    parser.add_argument(
        '--sdf_pial_dir', type=str, required=True,
        help='Path to SDF pial directory'
    )
    parser.add_argument(
        '--sdf_white_dir', type=str, required=True,
        help='Path to SDF white directory'
    )
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='Output directory for ribbon masks'
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Get list of SDF pial files
    file_list = sorted([
        f for f in os.listdir(args.sdf_pial_dir)
        if f.endswith('.nii.gz')
    ])

    LOG.info(f"Found {len(file_list)} SDF pial files")
    LOG.info(f"Output: {args.output_dir}")

    success = 0
    skipped = 0

    for filename in tqdm(file_list, desc="Generating Ribbon Masks"):
        sdf_pial_path = os.path.join(args.sdf_pial_dir, filename)
        sdf_white_path = os.path.join(args.sdf_white_dir, filename)

        if not os.path.exists(sdf_white_path):
            LOG.warning(f"[SKIP] {filename}: no matching white SDF")
            skipped += 1
            continue

        try:
            # Load SDFs
            sdf_pial_img = nib.load(sdf_pial_path)
            sdf_white_img = nib.load(sdf_white_path)

            sdf_pial_data = sdf_pial_img.get_fdata()
            sdf_white_data = sdf_white_img.get_fdata()

            # Generate ribbon mask
            ribbon_mask = cortical_ribbon_mask_generation(sdf_pial_data, sdf_white_data)

            # Save
            ribbon_img = nib.Nifti1Image(
                ribbon_mask.astype(np.float32),
                affine=sdf_pial_img.affine,
                header=sdf_pial_img.header
            )
            output_path = os.path.join(args.output_dir, filename)
            nib.save(ribbon_img, output_path)
            success += 1

        except Exception as e:
            LOG.error(f"[ERROR] {filename}: {e}")
            skipped += 1

    LOG.info(f"\n=== Summary ===")
    LOG.info(f"Total: {len(file_list)}, Success: {success}, Skipped: {skipped}")


if __name__ == '__main__':
    main()
