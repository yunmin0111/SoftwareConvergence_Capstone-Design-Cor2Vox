"""
Generate SDF, Edge Map, and Ribbon Mask ALL from ribbon.mgz only.
No surface mesh files needed — everything comes from one source.

Edge Map: extracted from label boundaries in ribbon.mgz
  - Pial edge:  boundary between 0 and {3, 42}
  - White edge: boundary between {3,42} and {2,41}
  - All edges combined into one binary volume

This ensures perfect alignment between all 4 data types.

Usage:
    python generate_all_from_ribbon.py \
        --fs_dir /data/datasets/CamCAN/freesurfer \
        --output_dir /data/yunmin0111/dataset_v3 \
        --num_workers 4
"""

import os
import argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt
from multiprocessing import Pool
from tqdm import tqdm
import logging
import traceback

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
LOG = logging.getLogger(__name__)


def generate_ribbon_mask(ribbon_data):
    """Cortex region from ribbon.mgz: labels 3 and 42."""
    mask = np.zeros_like(ribbon_data, dtype=np.float32)
    mask[(ribbon_data == 3) | (ribbon_data == 42)] = 1.0
    return mask


def compute_sdf(interior_mask):
    """SDF from binary interior mask: positive outside, negative inside."""
    dist_outside = distance_transform_edt(1 - interior_mask).astype(np.float32)
    dist_inside = distance_transform_edt(interior_mask).astype(np.float32)
    return dist_outside - dist_inside


def generate_sdf_pial(ribbon_data):
    """Inside pial = all brain tissue (2, 3, 41, 42)."""
    interior = np.zeros_like(ribbon_data, dtype=np.uint8)
    interior[(ribbon_data == 2) | (ribbon_data == 3) |
             (ribbon_data == 41) | (ribbon_data == 42)] = 1
    return compute_sdf(interior)


def generate_sdf_white(ribbon_data):
    """Inside white = white matter only (2, 41)."""
    interior = np.zeros_like(ribbon_data, dtype=np.uint8)
    interior[(ribbon_data == 2) | (ribbon_data == 41)] = 1
    return compute_sdf(interior)


def generate_edge_map_from_ribbon(ribbon_data):
    """
    Generate edge map by detecting label boundaries in ribbon.mgz.
    
    A voxel is an edge if any of its 6 neighbors has a different label.
    This captures all surfaces: pial (0↔3, 0↔42) and white (3↔2, 42↔41).
    
    Since ribbon.mgz is the same source as SDF and ribbon mask,
    edges are perfectly aligned with all other data.
    """
    H, W, D = ribbon_data.shape
    edge = np.zeros_like(ribbon_data, dtype=np.float32)
    
    # Only check voxels that are non-zero (brain tissue)
    # and their neighbors for label changes
    brain_mask = ribbon_data > 0
    
    # 6-connectivity: check each direction for label boundary
    # +x direction
    diff_x = ribbon_data[1:, :, :] != ribbon_data[:-1, :, :]
    edge[1:, :, :] |= diff_x.astype(np.uint8)
    edge[:-1, :, :] |= diff_x.astype(np.uint8)
    
    # +y direction  
    diff_y = ribbon_data[:, 1:, :] != ribbon_data[:, :-1, :]
    edge[:, 1:, :] |= diff_y.astype(np.uint8)
    edge[:, :-1, :] |= diff_y.astype(np.uint8)
    
    # +z direction
    diff_z = ribbon_data[:, :, 1:] != ribbon_data[:, :, :-1]
    edge[:, :, 1:] |= diff_z.astype(np.uint8)
    edge[:, :, :-1] |= diff_z.astype(np.uint8)
    
    # Only keep edges that involve brain tissue (exclude background-only boundaries)
    # At least one side of the boundary must be brain tissue
    edge = edge.astype(np.float32)
    
    # Remove edges between background voxels (far from brain)
    # Keep only edges where at least one neighbor is brain tissue
    brain_dilated = np.zeros_like(ribbon_data, dtype=np.uint8)
    brain_dilated[brain_mask] = 1
    # Dilate by 1 voxel
    brain_dilated[1:, :, :] |= brain_mask[:-1, :, :].astype(np.uint8)
    brain_dilated[:-1, :, :] |= brain_mask[1:, :, :].astype(np.uint8)
    brain_dilated[:, 1:, :] |= brain_mask[:, :-1, :].astype(np.uint8)
    brain_dilated[:, :-1, :] |= brain_mask[:, 1:, :].astype(np.uint8)
    brain_dilated[:, :, 1:] |= brain_mask[:, :, :-1].astype(np.uint8)
    brain_dilated[:, :, :-1] |= brain_mask[:, :, 1:].astype(np.uint8)
    
    edge = edge * brain_dilated.astype(np.float32)
    
    return edge


def process_subject(args):
    """Process one subject: generate all 4 data types from ribbon.mgz only."""
    subject_dir, out_pial, out_white, out_edge, out_ribbon = args
    subject_id = os.path.basename(subject_dir)

    try:
        mri_dir = os.path.join(subject_dir, 'mri')
        ribbon_path = os.path.join(mri_dir, 'ribbon.mgz')

        if not os.path.exists(ribbon_path):
            LOG.warning(f"[SKIP] {subject_id}: missing ribbon.mgz")
            return None

        # Skip if already processed
        out_files = [
            os.path.join(out_pial, f"{subject_id}.nii.gz"),
            os.path.join(out_white, f"{subject_id}.nii.gz"),
            os.path.join(out_edge, f"{subject_id}.nii.gz"),
            os.path.join(out_ribbon, f"{subject_id}.nii.gz"),
        ]
        if all(os.path.exists(f) for f in out_files):
            LOG.info(f"[SKIP] {subject_id}: already processed")
            return subject_id

        # Load ribbon.mgz
        ribbon_img = nib.load(ribbon_path)
        ribbon_data = ribbon_img.get_fdata()
        affine = ribbon_img.affine

        LOG.info(f"[{subject_id}] Processing from ribbon.mgz only...")

        # 1) SDF Pial
        sdf_pial = generate_sdf_pial(ribbon_data)
        nib.save(nib.Nifti1Image(sdf_pial, affine=affine), out_files[0])

        # 2) SDF White
        sdf_white = generate_sdf_white(ribbon_data)
        nib.save(nib.Nifti1Image(sdf_white, affine=affine), out_files[1])

        # 3) Edge Map (from ribbon.mgz label boundaries!)
        edge_map = generate_edge_map_from_ribbon(ribbon_data)
        nib.save(nib.Nifti1Image(edge_map, affine=affine), out_files[2])

        # 4) Ribbon Mask
        ribbon_mask = generate_ribbon_mask(ribbon_data)
        nib.save(nib.Nifti1Image(ribbon_mask, affine=affine), out_files[3])

        LOG.info(f"[DONE] {subject_id} | edge voxels: {int(edge_map.sum())}")
        return subject_id

    except Exception as e:
        LOG.error(f"[ERROR] {subject_id}: {e}")
        LOG.error(traceback.format_exc())
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate all preprocessing data from ribbon.mgz only"
    )
    parser.add_argument('--fs_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    out_pial = os.path.join(args.output_dir, 'sdf_pial')
    out_white = os.path.join(args.output_dir, 'sdf_white')
    out_edge = os.path.join(args.output_dir, 'edge_map')
    out_ribbon = os.path.join(args.output_dir, 'ribbon_mask')

    for d in [out_pial, out_white, out_edge, out_ribbon]:
        os.makedirs(d, exist_ok=True)

    subjects = sorted([
        os.path.join(args.fs_dir, d)
        for d in os.listdir(args.fs_dir)
        if os.path.isdir(os.path.join(args.fs_dir, d))
    ])

    LOG.info(f"Found {len(subjects)} subjects")
    LOG.info(f"Output: {args.output_dir}")
    LOG.info(f"ALL data generated from ribbon.mgz only!")

    task_args = [
        (subj, out_pial, out_white, out_edge, out_ribbon)
        for subj in subjects
    ]

    if args.num_workers > 1:
        with Pool(args.num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_subject, task_args),
                total=len(task_args),
                desc="Processing"
            ))
    else:
        results = []
        for task in tqdm(task_args, desc="Processing"):
            results.append(process_subject(task))

    successful = [r for r in results if r is not None]
    LOG.info(f"\n=== Summary ===")
    LOG.info(f"Total: {len(results)}")
    LOG.info(f"Successful: {len(successful)}")
    LOG.info(f"Failed: {len(results) - len(successful)}")

    with open(os.path.join(args.output_dir, 'valid_subjects.txt'), 'w') as f:
        for s in sorted(successful):
            f.write(s + '\n')


if __name__ == '__main__':
    main()
