"""
Generate SDF (Signed Distance Function) volumes from FreeSurfer cortical surfaces.

Converts FreeSurfer surface meshes (lh.pial, rh.pial, lh.white, rh.white)
into 3D SDF volumes (.nii.gz) for Cor2Vox training.

Input:
    - FreeSurfer subjects directory containing surf/ and mri/ folders
Output:
    - sdf_pial/*.nii.gz   (one per subject)
    - sdf_white/*.nii.gz  (one per subject)

Usage:
    python generate_sdf.py \
        --fs_dir /data/datasets/CamCAN/freesurfer \
        --output_dir /data/yunmin0111/dataset \
        --resolution 256 \
        --num_workers 4
"""

import os
import argparse
import numpy as np
import nibabel as nib
import trimesh
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


def surface_to_volume_mask(vertices, faces, vol_shape, affine):
    """
    Rasterize a surface mesh into a binary volume.
    Voxels inside the mesh are set to 1, outside to 0.

    Args:
        vertices: (N, 3) array of vertex coordinates in world (RAS) space
        faces: (M, 3) array of triangle face indices
        vol_shape: tuple (H, W, D) shape of the output volume
        affine: (4, 4) voxel-to-world affine matrix

    Returns:
        mask: (H, W, D) binary numpy array
    """
    # Convert world coordinates to voxel coordinates
    inv_affine = np.linalg.inv(affine)
    voxel_coords = nib.affines.apply_affine(inv_affine, vertices)

    # Create trimesh object in voxel space
    mesh = trimesh.Trimesh(vertices=voxel_coords, faces=faces, process=False)

    # Create a voxel grid and check which voxels are inside the mesh
    # Use ray-based method for inside/outside determination
    H, W, D = vol_shape
    mask = np.zeros(vol_shape, dtype=np.uint8)

    # Generate grid points (center of each voxel)
    grid_points = np.mgrid[0:H, 0:W, 0:D].reshape(3, -1).T.astype(np.float64) + 0.5

    # Check containment in batches to avoid memory issues
    batch_size = 100000
    inside = np.zeros(len(grid_points), dtype=bool)

    for start in range(0, len(grid_points), batch_size):
        end = min(start + batch_size, len(grid_points))
        batch = grid_points[start:end]
        inside[start:end] = mesh.contains(batch)

    mask = inside.reshape(vol_shape).astype(np.uint8)

    return mask


def compute_sdf_from_mask(mask):
    """
    Compute a signed distance function from a binary mask.
    Positive values are outside the surface, negative values are inside.

    Args:
        mask: (H, W, D) binary numpy array (1=inside, 0=outside)

    Returns:
        sdf: (H, W, D) numpy float32 array
    """
    # Distance transform for outside (where mask == 0)
    dist_outside = distance_transform_edt(1 - mask).astype(np.float32)

    # Distance transform for inside (where mask == 1)
    dist_inside = distance_transform_edt(mask).astype(np.float32)

    # SDF: positive outside, negative inside
    sdf = dist_outside - dist_inside

    return sdf


def merge_hemisphere_sdfs(sdf_lh, sdf_rh):
    """
    Merge left and right hemisphere SDFs into a single SDF volume.
    Uses the minimum absolute distance (union of surfaces).

    Args:
        sdf_lh: (H, W, D) SDF for left hemisphere
        sdf_rh: (H, W, D) SDF for right hemisphere

    Returns:
        sdf_merged: (H, W, D) merged SDF
    """
    # For union: take the minimum of the two SDFs
    # Where both are positive (outside both), take the smaller positive value
    # Where both are negative (inside both), take the larger (less negative) value
    # Where signs differ, take the negative one (inside at least one surface)
    sdf_merged = np.where(
        np.abs(sdf_lh) < np.abs(sdf_rh),
        sdf_lh,
        sdf_rh
    )
    return sdf_merged


def process_subject(args):
    """
    Process a single subject: generate SDF volumes for pial and white surfaces.

    Args:
        args: tuple of (subject_dir, output_pial_dir, output_white_dir, resolution)

    Returns:
        subject_id: string ID of the subject, or None if failed
    """
    subject_dir, output_pial_dir, output_white_dir, resolution = args

    subject_id = os.path.basename(subject_dir)

    try:
        # --- Check required files exist ---
        surf_dir = os.path.join(subject_dir, 'surf')
        mri_dir = os.path.join(subject_dir, 'mri')

        required_surfs = ['lh.pial', 'rh.pial', 'lh.white', 'rh.white']
        for s in required_surfs:
            if not os.path.exists(os.path.join(surf_dir, s)):
                LOG.warning(f"[SKIP] {subject_id}: missing {s}")
                return None

        ref_mgz = os.path.join(mri_dir, 'T1.mgz')
        if not os.path.exists(ref_mgz):
            LOG.warning(f"[SKIP] {subject_id}: missing T1.mgz")
            return None

        # --- Load reference volume for affine and shape ---
        ref_img = nib.load(ref_mgz)
        affine = ref_img.affine
        vol_shape = ref_img.shape[:3]  # (256, 256, 256)

        # --- Generate Pial SDF ---
        lh_pial_v, lh_pial_f = nib.freesurfer.read_geometry(
            os.path.join(surf_dir, 'lh.pial')
        )
        rh_pial_v, rh_pial_f = nib.freesurfer.read_geometry(
            os.path.join(surf_dir, 'rh.pial')
        )

        LOG.info(f"[{subject_id}] Computing pial SDF...")
        lh_pial_mask = surface_to_volume_mask(lh_pial_v, lh_pial_f, vol_shape, affine)
        rh_pial_mask = surface_to_volume_mask(rh_pial_v, rh_pial_f, vol_shape, affine)

        lh_pial_sdf = compute_sdf_from_mask(lh_pial_mask)
        rh_pial_sdf = compute_sdf_from_mask(rh_pial_mask)
        pial_sdf = merge_hemisphere_sdfs(lh_pial_sdf, rh_pial_sdf)

        # Save pial SDF
        out_pial = os.path.join(output_pial_dir, f"{subject_id}.nii.gz")
        nib.save(nib.Nifti1Image(pial_sdf, affine=affine), out_pial)

        # --- Generate White SDF ---
        lh_white_v, lh_white_f = nib.freesurfer.read_geometry(
            os.path.join(surf_dir, 'lh.white')
        )
        rh_white_v, rh_white_f = nib.freesurfer.read_geometry(
            os.path.join(surf_dir, 'rh.white')
        )

        LOG.info(f"[{subject_id}] Computing white SDF...")
        lh_white_mask = surface_to_volume_mask(lh_white_v, lh_white_f, vol_shape, affine)
        rh_white_mask = surface_to_volume_mask(rh_white_v, rh_white_f, vol_shape, affine)

        lh_white_sdf = compute_sdf_from_mask(lh_white_mask)
        rh_white_sdf = compute_sdf_from_mask(rh_white_mask)
        white_sdf = merge_hemisphere_sdfs(lh_white_sdf, rh_white_sdf)

        # Save white SDF
        out_white = os.path.join(output_white_dir, f"{subject_id}.nii.gz")
        nib.save(nib.Nifti1Image(white_sdf, affine=affine), out_white)

        LOG.info(f"[DONE] {subject_id}")
        return subject_id

    except Exception as e:
        LOG.error(f"[ERROR] {subject_id}: {e}")
        LOG.error(traceback.format_exc())
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate SDF volumes from FreeSurfer cortical surfaces"
    )
    parser.add_argument(
        '--fs_dir', type=str, required=True,
        help='Path to FreeSurfer subjects directory '
             '(e.g., /data/datasets/CamCAN/freesurfer)'
    )
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='Output directory for processed data '
             '(e.g., /data/yunmin0111/dataset)'
    )
    parser.add_argument(
        '--resolution', type=int, default=256,
        help='Volume resolution (default: 256, will be resized to 128 during training)'
    )
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='Number of parallel workers (default: 4)'
    )
    args = parser.parse_args()

    # Create output directories
    output_pial_dir = os.path.join(args.output_dir, 'sdf_pial')
    output_white_dir = os.path.join(args.output_dir, 'sdf_white')
    os.makedirs(output_pial_dir, exist_ok=True)
    os.makedirs(output_white_dir, exist_ok=True)

    # Get list of subjects
    subjects = sorted([
        os.path.join(args.fs_dir, d)
        for d in os.listdir(args.fs_dir)
        if os.path.isdir(os.path.join(args.fs_dir, d))
    ])

    LOG.info(f"Found {len(subjects)} subjects in {args.fs_dir}")
    LOG.info(f"Output pial SDF: {output_pial_dir}")
    LOG.info(f"Output white SDF: {output_white_dir}")

    # Prepare arguments for parallel processing
    task_args = [
        (subj, output_pial_dir, output_white_dir, args.resolution)
        for subj in subjects
    ]

    # Process subjects
    if args.num_workers > 1:
        with Pool(args.num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_subject, task_args),
                total=len(task_args),
                desc="Generating SDFs"
            ))
    else:
        results = []
        for task in tqdm(task_args, desc="Generating SDFs"):
            results.append(process_subject(task))

    # Summary
    successful = [r for r in results if r is not None]
    failed = len(results) - len(successful)
    LOG.info(f"\n=== Summary ===")
    LOG.info(f"Total subjects: {len(results)}")
    LOG.info(f"Successful: {len(successful)}")
    LOG.info(f"Failed/Skipped: {failed}")

    # Save list of successful subjects
    id_list_path = os.path.join(args.output_dir, 'valid_subjects.txt')
    with open(id_list_path, 'w') as f:
        for s in sorted(successful):
            f.write(s + '\n')
    LOG.info(f"Valid subject list saved to {id_list_path}")


if __name__ == '__main__':
    main()
