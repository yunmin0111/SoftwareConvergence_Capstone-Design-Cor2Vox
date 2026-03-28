"""
Generate SDF, Edge Map, and Ribbon Mask using FreeSurfer's native files.
Uses ribbon.mgz, filled.mgz, and surface meshes for accurate data.

This version uses FreeSurfer's own segmentation results instead of
custom rasterization, producing much more accurate outputs.

Input:
    FreeSurfer subjects directory containing:
    - mri/ribbon.mgz      (cortical ribbon labels)
    - mri/filled.mgz       (filled white matter volume)
    - mri/T1.mgz           (reference volume)
    - mri/brain.mgz        (skull-stripped brain)
    - surf/lh.pial, rh.pial, lh.white, rh.white (surface meshes)

Output:
    - sdf_pial/*.nii.gz     (SDF from pial surface, using ribbon.mgz for inside/outside)
    - sdf_white/*.nii.gz    (SDF from white surface, using filled.mgz for inside/outside)
    - edge_map/*.nii.gz     (edge from surface meshes via Bresenham)
    - ribbon_mask/*.nii.gz  (directly from ribbon.mgz)

Usage:
    python generate_all_native.py \
        --fs_dir /data/datasets/CamCAN/freesurfer \
        --output_dir /data/yunmin0111/dataset_v2 \
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


# ============================================================
# Ribbon Mask: directly from FreeSurfer's ribbon.mgz
# ============================================================
def generate_ribbon_mask(ribbon_data):
    """
    Generate cortical ribbon mask from FreeSurfer's ribbon.mgz.

    ribbon.mgz values:
        0  = background
        2  = Left hemisphere white matter
        3  = Left hemisphere cortex (pial)
        41 = Right hemisphere white matter
        42 = Right hemisphere cortex (pial)

    Cortical ribbon = labels 3 and 42 (cortex regions)

    Returns:
        mask: (256, 256, 256) binary mask, 1 = cortex
    """
    mask = np.zeros_like(ribbon_data, dtype=np.float32)
    mask[(ribbon_data == 3) | (ribbon_data == 42)] = 1.0
    return mask


# ============================================================
# SDF: using FreeSurfer's segmentation for inside/outside
# ============================================================
def compute_sdf_from_segmentation(interior_mask):
    """
    Compute SDF from a binary interior mask.

    Uses distance_transform_edt for both inside and outside distances.
    Surface = boundary between interior and exterior.

    Args:
        interior_mask: (H, W, D) binary mask (1 = inside surface)

    Returns:
        sdf: (H, W, D) float32, negative inside, positive outside, 0 at boundary
    """
    # Distance from boundary for outside voxels
    dist_outside = distance_transform_edt(1 - interior_mask).astype(np.float32)

    # Distance from boundary for inside voxels
    dist_inside = distance_transform_edt(interior_mask).astype(np.float32)

    # SDF: positive outside, negative inside
    sdf = dist_outside - dist_inside

    return sdf


def generate_sdf_pial(ribbon_data):
    """
    Generate SDF for pial surface using ribbon.mgz.

    Everything inside the pial surface = cortex(3,42) + white matter(2,41)
    Everything outside = background(0)

    Returns:
        sdf: (256, 256, 256) float32
    """
    # Inside pial = all brain tissue (cortex + white matter)
    interior = np.zeros_like(ribbon_data, dtype=np.uint8)
    interior[(ribbon_data == 2) | (ribbon_data == 3) |
             (ribbon_data == 41) | (ribbon_data == 42)] = 1

    sdf = compute_sdf_from_segmentation(interior)
    return sdf


def generate_sdf_white(ribbon_data):
    """
    Generate SDF for white matter surface using ribbon.mgz.

    Everything inside the white surface = white matter only (2, 41)
    Everything outside white surface = cortex(3,42) + background(0)

    Returns:
        sdf: (256, 256, 256) float32
    """
    # Inside white = white matter only
    interior = np.zeros_like(ribbon_data, dtype=np.uint8)
    interior[(ribbon_data == 2) | (ribbon_data == 41)] = 1

    sdf = compute_sdf_from_segmentation(interior)
    return sdf


# ============================================================
# Edge Map: from surface meshes via Bresenham
# ============================================================
def bresenham_3d(p0, p1):
    """3D Bresenham's line algorithm."""
    x0, y0, z0 = p0
    x1, y1, z1 = p1
    dx, dy, dz = abs(x1-x0), abs(y1-y0), abs(z1-z0)
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    sz = 1 if z1 > z0 else -1

    if dx >= dy and dx >= dz:
        ey, ez = 2*dy - dx, 2*dz - dx
        points = []
        x, y, z = x0, y0, z0
        for _ in range(dx + 1):
            points.append((x, y, z))
            if ey >= 0: y += sy; ey -= 2*dx
            if ez >= 0: z += sz; ez -= 2*dx
            ey += 2*dy; ez += 2*dz; x += sx
    elif dy >= dx and dy >= dz:
        ex, ez = 2*dx - dy, 2*dz - dy
        points = []
        x, y, z = x0, y0, z0
        for _ in range(dy + 1):
            points.append((x, y, z))
            if ex >= 0: x += sx; ex -= 2*dy
            if ez >= 0: z += sz; ez -= 2*dy
            ex += 2*dx; ez += 2*dz; y += sy
    else:
        ex, ey = 2*dx - dz, 2*dy - dz
        points = []
        x, y, z = x0, y0, z0
        for _ in range(dz + 1):
            points.append((x, y, z))
            if ex >= 0: x += sx; ex -= 2*dz
            if ey >= 0: y += sy; ey -= 2*dz
            ex += 2*dx; ey += 2*dy; z += sz
    return points


def generate_edge_map(surf_dir, vol_shape, affine):
    """
    Generate edge map from all 4 surface meshes.
    Uses Bresenham to draw mesh edges on voxel grid.

    Args:
        surf_dir: path to FreeSurfer surf/ directory
        vol_shape: (H, W, D)
        affine: voxel-to-world affine matrix

    Returns:
        edge_vol: (H, W, D) float32 binary volume
    """
    H, W, D = vol_shape
    edge_vol = np.zeros(vol_shape, dtype=np.uint8)
    inv_affine = np.linalg.inv(affine)

    surfaces = ['lh.pial', 'rh.pial', 'lh.white', 'rh.white']

    for surf_name in surfaces:
        surf_path = os.path.join(surf_dir, surf_name)
        if not os.path.exists(surf_path):
            LOG.warning(f"  Missing {surf_name}, skipping")
            continue

        verts, faces = nib.freesurfer.read_geometry(surf_path)
        voxel_coords = nib.affines.apply_affine(inv_affine, verts)
        voxel_int = np.round(voxel_coords).astype(np.int32)

        # Extract unique edges
        edges = np.concatenate([
            faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]
        ], axis=0)
        edges_sorted = np.sort(edges, axis=1)
        edges_unique = np.unique(edges_sorted, axis=0)

        for e in edges_unique:
            p0, p1 = voxel_int[e[0]], voxel_int[e[1]]
            points = bresenham_3d(p0, p1)
            for x, y, z in points:
                if 0 <= x < H and 0 <= y < W and 0 <= z < D:
                    edge_vol[x, y, z] = 1

    return edge_vol.astype(np.float32)


# ============================================================
# Process single subject
# ============================================================
def process_subject(args):
    """Process one subject: generate all 4 data types."""
    subject_dir, out_pial, out_white, out_edge, out_ribbon = args
    subject_id = os.path.basename(subject_dir)

    try:
        mri_dir = os.path.join(subject_dir, 'mri')
        surf_dir = os.path.join(subject_dir, 'surf')

        # Check required files
        ribbon_path = os.path.join(mri_dir, 'ribbon.mgz')
        t1_path = os.path.join(mri_dir, 'T1.mgz')

        if not os.path.exists(ribbon_path):
            LOG.warning(f"[SKIP] {subject_id}: missing ribbon.mgz")
            return None
        if not os.path.exists(t1_path):
            LOG.warning(f"[SKIP] {subject_id}: missing T1.mgz")
            return None

        required_surfs = ['lh.pial', 'rh.pial', 'lh.white', 'rh.white']
        for s in required_surfs:
            if not os.path.exists(os.path.join(surf_dir, s)):
                LOG.warning(f"[SKIP] {subject_id}: missing {s}")
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

        # Load ribbon.mgz and T1.mgz
        ribbon_img = nib.load(ribbon_path)
        ribbon_data = ribbon_img.get_fdata()
        affine = ribbon_img.affine
        vol_shape = ribbon_data.shape[:3]

        LOG.info(f"[{subject_id}] Generating from ribbon.mgz...")

        # 1) Ribbon Mask
        ribbon_mask = generate_ribbon_mask(ribbon_data)
        nib.save(
            nib.Nifti1Image(ribbon_mask, affine=affine),
            out_files[3]
        )

        # 2) SDF Pial
        sdf_pial = generate_sdf_pial(ribbon_data)
        nib.save(
            nib.Nifti1Image(sdf_pial, affine=affine),
            out_files[0]
        )

        # 3) SDF White
        sdf_white = generate_sdf_white(ribbon_data)
        nib.save(
            nib.Nifti1Image(sdf_white, affine=affine),
            out_files[1]
        )

        # 4) Edge Map (still needs surface meshes)
        LOG.info(f"[{subject_id}] Generating edge map...")
        edge_map = generate_edge_map(surf_dir, vol_shape, affine)
        nib.save(
            nib.Nifti1Image(edge_map, affine=affine),
            out_files[2]
        )

        LOG.info(f"[DONE] {subject_id}")
        return subject_id

    except Exception as e:
        LOG.error(f"[ERROR] {subject_id}: {e}")
        LOG.error(traceback.format_exc())
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate SDF/Edge/Ribbon from FreeSurfer native files"
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
