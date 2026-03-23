"""
Generate Edge Map volumes from FreeSurfer cortical surfaces.

Rasterizes the edges of pial and white matter surface meshes into
a 3D binary volume (.nii.gz) for Cor2Vox training.

The edge map represents the boundary contours of cortical surfaces
in voxel space, used as an auxiliary condition in the Brownian bridge
diffusion process.

Input:
    - FreeSurfer subjects directory containing surf/ and mri/ folders
Output:
    - edge_map/*.nii.gz   (one per subject)

Usage:
    python generate_edge_map.py \
        --fs_dir /data/datasets/CamCAN/freesurfer \
        --output_dir /data/yunmin0111/dataset \
        --num_workers 4
"""

import os
import argparse
import numpy as np
import nibabel as nib
from multiprocessing import Pool
from tqdm import tqdm
import logging
import traceback

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
LOG = logging.getLogger(__name__)


def rasterize_edges(vertices, faces, vol_shape, affine):
    """
    Rasterize mesh edges into a binary 3D volume using Bresenham-like
    line drawing between connected vertices.

    Args:
        vertices: (N, 3) vertex coordinates in world (RAS) space
        faces: (M, 3) triangle face indices
        vol_shape: tuple (H, W, D)
        affine: (4, 4) voxel-to-world affine matrix

    Returns:
        edge_vol: (H, W, D) binary numpy array
    """
    # Convert world coordinates to voxel coordinates
    inv_affine = np.linalg.inv(affine)
    voxel_coords = nib.affines.apply_affine(inv_affine, vertices)

    # Round to nearest integer voxel
    voxel_int = np.round(voxel_coords).astype(np.int32)

    H, W, D = vol_shape
    edge_vol = np.zeros(vol_shape, dtype=np.uint8)

    # Extract unique edges from faces
    # Each face (v0, v1, v2) has edges: (v0,v1), (v1,v2), (v2,v0)
    edges = np.concatenate([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]]
    ], axis=0)

    # Remove duplicate edges (sort each edge pair first)
    edges_sorted = np.sort(edges, axis=1)
    edges_unique = np.unique(edges_sorted, axis=0)

    # Draw lines between connected vertices using 3D Bresenham
    for e in edges_unique:
        p0 = voxel_int[e[0]]
        p1 = voxel_int[e[1]]
        points = bresenham_3d(p0, p1)

        for p in points:
            x, y, z = p
            if 0 <= x < H and 0 <= y < W and 0 <= z < D:
                edge_vol[x, y, z] = 1

    return edge_vol


def bresenham_3d(p0, p1):
    """
    3D Bresenham's line algorithm.
    Returns list of (x, y, z) integer coordinates along the line.

    Args:
        p0: (3,) start point
        p1: (3,) end point

    Returns:
        points: list of (x, y, z) tuples
    """
    x0, y0, z0 = p0
    x1, y1, z1 = p1

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    dz = abs(z1 - z0)

    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    sz = 1 if z1 > z0 else -1

    # Driving axis is the one with the largest delta
    if dx >= dy and dx >= dz:
        # X is driving axis
        ey = 2 * dy - dx
        ez = 2 * dz - dx
        points = []
        x, y, z = x0, y0, z0
        for _ in range(dx + 1):
            points.append((x, y, z))
            if ey >= 0:
                y += sy
                ey -= 2 * dx
            if ez >= 0:
                z += sz
                ez -= 2 * dx
            ey += 2 * dy
            ez += 2 * dz
            x += sx
    elif dy >= dx and dy >= dz:
        # Y is driving axis
        ex = 2 * dx - dy
        ez = 2 * dz - dy
        points = []
        x, y, z = x0, y0, z0
        for _ in range(dy + 1):
            points.append((x, y, z))
            if ex >= 0:
                x += sx
                ex -= 2 * dy
            if ez >= 0:
                z += sz
                ez -= 2 * dy
            ex += 2 * dx
            ez += 2 * dz
            y += sy
    else:
        # Z is driving axis
        ex = 2 * dx - dz
        ey = 2 * dy - dz
        points = []
        x, y, z = x0, y0, z0
        for _ in range(dz + 1):
            points.append((x, y, z))
            if ex >= 0:
                x += sx
                ex -= 2 * dz
            if ey >= 0:
                y += sy
                ey -= 2 * dz
            ex += 2 * dx
            ey += 2 * dy
            z += sz

    return points


def process_subject(args):
    """
    Process a single subject: generate edge map volume.

    Args:
        args: tuple of (subject_dir, output_dir)

    Returns:
        subject_id: string ID, or None if failed
    """
    subject_dir, output_dir = args
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

        # Load reference volume
        ref_img = nib.load(ref_mgz)
        affine = ref_img.affine
        vol_shape = ref_img.shape[:3]

        # Initialize combined edge volume
        edge_vol = np.zeros(vol_shape, dtype=np.uint8)

        # Process all 4 surfaces
        for surf_name in required_surfs:
            LOG.info(f"[{subject_id}] Rasterizing edges: {surf_name}")
            verts, faces = nib.freesurfer.read_geometry(
                os.path.join(surf_dir, surf_name)
            )
            surf_edges = rasterize_edges(verts, faces, vol_shape, affine)
            edge_vol = np.maximum(edge_vol, surf_edges)

        # Save edge map
        out_path = os.path.join(output_dir, f"{subject_id}.nii.gz")
        nib.save(nib.Nifti1Image(edge_vol.astype(np.float32), affine=affine), out_path)

        LOG.info(f"[DONE] {subject_id}")
        return subject_id

    except Exception as e:
        LOG.error(f"[ERROR] {subject_id}: {e}")
        LOG.error(traceback.format_exc())
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate Edge Map volumes from FreeSurfer cortical surfaces"
    )
    parser.add_argument(
        '--fs_dir', type=str, required=True,
        help='Path to FreeSurfer subjects directory'
    )
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='Output base directory (edge_map/ will be created inside)'
    )
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='Number of parallel workers (default: 4)'
    )
    args = parser.parse_args()

    output_edge_dir = os.path.join(args.output_dir, 'edge_map')
    os.makedirs(output_edge_dir, exist_ok=True)

    # Get list of subjects
    subjects = sorted([
        os.path.join(args.fs_dir, d)
        for d in os.listdir(args.fs_dir)
        if os.path.isdir(os.path.join(args.fs_dir, d))
    ])

    LOG.info(f"Found {len(subjects)} subjects in {args.fs_dir}")
    LOG.info(f"Output edge map: {output_edge_dir}")

    task_args = [(subj, output_edge_dir) for subj in subjects]

    if args.num_workers > 1:
        with Pool(args.num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_subject, task_args),
                total=len(task_args),
                desc="Generating Edge Maps"
            ))
    else:
        results = []
        for task in tqdm(task_args, desc="Generating Edge Maps"):
            results.append(process_subject(task))

    successful = [r for r in results if r is not None]
    failed = len(results) - len(successful)
    LOG.info(f"\n=== Summary ===")
    LOG.info(f"Total: {len(results)}, Success: {len(successful)}, Failed: {failed}")


if __name__ == '__main__':
    main()
