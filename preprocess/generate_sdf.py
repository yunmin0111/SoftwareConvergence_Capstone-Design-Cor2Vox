"""
Generate SDF (Signed Distance Function) volumes from FreeSurfer cortical surfaces.
FAST VERSION - uses surface rasterization + flood fill + EDT
(replaces slow trimesh.contains() method)

Converts FreeSurfer surface meshes (lh.pial, rh.pial, lh.white, rh.white)
into 3D SDF volumes (.nii.gz) for Cor2Vox training.

Method:
    1. Rasterize surface triangles into a thin shell in voxel space
    2. Flood fill (binary_fill_holes) to find interior region
    3. Compute EDT from shell for inside/outside
    4. SDF = outside_distance - inside_distance

Speed: ~10-30 sec per subject (vs hours with trimesh.contains)

Input:
    - FreeSurfer subjects directory containing surf/ and mri/ folders
Output:
    - sdf_pial/*.nii.gz   (one per subject)
    - sdf_white/*.nii.gz  (one per subject)

Usage:
    python generate_sdf.py \
        --fs_dir /data/datasets/CamCAN/freesurfer \
        --output_dir /data/yunmin0111/dataset \
        --num_workers 4
"""

import os
import argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt, binary_fill_holes
from multiprocessing import Pool
from tqdm import tqdm
import logging
import traceback

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
LOG = logging.getLogger(__name__)


def rasterize_triangles_fast(vertices, faces, vol_shape):
    """
    Fast triangle rasterization: fill bounding box of each triangle.
    Creates a thin shell volume marking surface voxels.

    Args:
        vertices: (N, 3) voxel-space coordinates (float)
        faces: (M, 3) triangle indices
        vol_shape: (H, W, D)

    Returns:
        shell: (H, W, D) binary volume (uint8)
    """
    H, W, D = vol_shape
    shell = np.zeros(vol_shape, dtype=np.uint8)

    tri_verts = vertices[faces]  # (M, 3, 3)

    # Compute bounding boxes for all triangles at once
    tri_min = np.floor(tri_verts.min(axis=1)).astype(np.int32)  # (M, 3)
    tri_max = np.ceil(tri_verts.max(axis=1)).astype(np.int32)   # (M, 3)

    # Clamp to volume bounds
    tri_min = np.maximum(tri_min, 0)
    tri_max[:, 0] = np.minimum(tri_max[:, 0], H - 1)
    tri_max[:, 1] = np.minimum(tri_max[:, 1], W - 1)
    tri_max[:, 2] = np.minimum(tri_max[:, 2], D - 1)

    # Fill bounding box of each triangle
    for i in range(len(faces)):
        x0, y0, z0 = tri_min[i]
        x1, y1, z1 = tri_max[i]
        shell[x0:x1+1, y0:y1+1, z0:z1+1] = 1

    return shell


def compute_sdf(vertices, faces, vol_shape):
    """
    Compute SDF for a single surface mesh using fast method.

    Steps:
        1. Rasterize triangles to get surface shell
        2. binary_fill_holes to determine interior
        3. EDT from shell boundary
        4. SDF = positive outside, negative inside

    Args:
        vertices: (N, 3) voxel-space coordinates
        faces: (M, 3) face indices
        vol_shape: (H, W, D)

    Returns:
        sdf: (H, W, D) float32 array
    """
    # Step 1: Rasterize surface into shell
    shell = rasterize_triangles_fast(vertices, faces, vol_shape)

    # Step 2: Fill holes to find interior
    # binary_fill_holes: everything unreachable from outside = interior
    filled = binary_fill_holes(shell).astype(np.uint8)

    # Step 3: Compute distance from surface shell
    # EDT of (1 - shell) gives distance to nearest shell voxel
    dist_from_surface = distance_transform_edt(1 - shell).astype(np.float32)

    # Step 4: Assign sign based on inside/outside
    # filled=1 and shell=0 means interior -> negative distance
    interior = (filled == 1) & (shell == 0)
    sdf = dist_from_surface.copy()
    sdf[interior] = -sdf[interior]
    sdf[shell == 1] = 0.0

    return sdf


def merge_hemisphere_sdfs(sdf_lh, sdf_rh):
    """
    Merge left and right hemisphere SDFs (union).
    Takes the SDF with smaller absolute value at each voxel.
    """
    return np.where(
        np.abs(sdf_lh) < np.abs(sdf_rh),
        sdf_lh,
        sdf_rh
    )


def process_subject(args):
    """
    Process a single subject: generate pial and white SDF volumes.

    Returns:
        subject_id or None if failed/skipped
    """
    subject_dir, output_pial_dir, output_white_dir = args
    subject_id = os.path.basename(subject_dir)

    try:
        surf_dir = os.path.join(subject_dir, 'surf')
        mri_dir = os.path.join(subject_dir, 'mri')

        # Check required files
        required_surfs = ['lh.pial', 'rh.pial', 'lh.white', 'rh.white']
        for s in required_surfs:
            if not os.path.exists(os.path.join(surf_dir, s)):
                LOG.warning(f"[SKIP] {subject_id}: missing {s}")
                return None

        ref_mgz = os.path.join(mri_dir, 'T1.mgz')
        if not os.path.exists(ref_mgz):
            LOG.warning(f"[SKIP] {subject_id}: missing T1.mgz")
            return None

        # Skip if already processed
        out_pial = os.path.join(output_pial_dir, f"{subject_id}.nii.gz")
        out_white = os.path.join(output_white_dir, f"{subject_id}.nii.gz")
        if os.path.exists(out_pial) and os.path.exists(out_white):
            LOG.info(f"[SKIP] {subject_id}: already processed")
            return subject_id

        # Load reference volume for affine and shape
        ref_img = nib.load(ref_mgz)
        affine = ref_img.affine
        vol_shape = ref_img.shape[:3]  # (256, 256, 256)
        inv_affine = np.linalg.inv(affine)

        # ===== Pial SDF =====
        LOG.info(f"[{subject_id}] Computing pial SDF...")

        lh_pial_v, lh_pial_f = nib.freesurfer.read_geometry(
            os.path.join(surf_dir, 'lh.pial')
        )
        rh_pial_v, rh_pial_f = nib.freesurfer.read_geometry(
            os.path.join(surf_dir, 'rh.pial')
        )

        # Convert world (RAS) to voxel coordinates
        lh_pial_vox = nib.affines.apply_affine(inv_affine, lh_pial_v)
        rh_pial_vox = nib.affines.apply_affine(inv_affine, rh_pial_v)

        lh_pial_sdf = compute_sdf(lh_pial_vox, lh_pial_f, vol_shape)
        rh_pial_sdf = compute_sdf(rh_pial_vox, rh_pial_f, vol_shape)
        pial_sdf = merge_hemisphere_sdfs(lh_pial_sdf, rh_pial_sdf)

        nib.save(nib.Nifti1Image(pial_sdf, affine=affine), out_pial)

        # ===== White SDF =====
        LOG.info(f"[{subject_id}] Computing white SDF...")

        lh_white_v, lh_white_f = nib.freesurfer.read_geometry(
            os.path.join(surf_dir, 'lh.white')
        )
        rh_white_v, rh_white_f = nib.freesurfer.read_geometry(
            os.path.join(surf_dir, 'rh.white')
        )

        lh_white_vox = nib.affines.apply_affine(inv_affine, lh_white_v)
        rh_white_vox = nib.affines.apply_affine(inv_affine, rh_white_v)

        lh_white_sdf = compute_sdf(lh_white_vox, lh_white_f, vol_shape)
        rh_white_sdf = compute_sdf(rh_white_vox, rh_white_f, vol_shape)
        white_sdf = merge_hemisphere_sdfs(lh_white_sdf, rh_white_sdf)

        nib.save(nib.Nifti1Image(white_sdf, affine=affine), out_white)

        LOG.info(f"[DONE] {subject_id}")
        return subject_id

    except Exception as e:
        LOG.error(f"[ERROR] {subject_id}: {e}")
        LOG.error(traceback.format_exc())
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate SDF volumes from FreeSurfer cortical surfaces (FAST)"
    )
    parser.add_argument(
        '--fs_dir', type=str, required=True,
        help='Path to FreeSurfer subjects directory'
    )
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='Output directory for processed data'
    )
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='Number of parallel workers (default: 4)'
    )
    args = parser.parse_args()

    output_pial_dir = os.path.join(args.output_dir, 'sdf_pial')
    output_white_dir = os.path.join(args.output_dir, 'sdf_white')
    os.makedirs(output_pial_dir, exist_ok=True)
    os.makedirs(output_white_dir, exist_ok=True)

    subjects = sorted([
        os.path.join(args.fs_dir, d)
        for d in os.listdir(args.fs_dir)
        if os.path.isdir(os.path.join(args.fs_dir, d))
    ])

    LOG.info(f"Found {len(subjects)} subjects in {args.fs_dir}")
    LOG.info(f"Output pial SDF: {output_pial_dir}")
    LOG.info(f"Output white SDF: {output_white_dir}")

    task_args = [
        (subj, output_pial_dir, output_white_dir)
        for subj in subjects
    ]

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

    successful = [r for r in results if r is not None]
    failed = len(results) - len(successful)
    LOG.info(f"\n=== Summary ===")
    LOG.info(f"Total: {len(results)}")
    LOG.info(f"Successful: {len(successful)}")
    LOG.info(f"Failed/Skipped: {failed}")

    id_list_path = os.path.join(args.output_dir, 'valid_subjects.txt')
    with open(id_list_path, 'w') as f:
        for s in sorted(successful):
            f.write(s + '\n')
    LOG.info(f"Valid subject list saved to {id_list_path}")


if __name__ == '__main__':
    main()
