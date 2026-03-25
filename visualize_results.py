"""
Visualize Cor2Vox sampling results.

Generates 2D slice images (axial, coronal, sagittal) and 3D mesh files
(.obj, .stl) from synthesized MRI volumes.

Input:
    - sample_to_eval/ folder containing .nii.gz files from Cor2Vox test
Output:
    - 2D PNG images (axial, coronal, sagittal slices for both real and synthetic)
    - 3D OBJ/STL mesh files
    - Comparison images (real vs synthetic side by side)

Usage:
    python visualize_results.py \
        --input_dir /data/yunmin0111/Cor2Vox/results/c2v/c2v/sample_to_eval/Test_samples_c2v \
        --output_dir /data/yunmin0111/visualizations
"""

import os
import argparse
import numpy as np
import nibabel as nib
from PIL import Image, ImageDraw, ImageFont
from skimage import measure
import trimesh
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
LOG = logging.getLogger(__name__)


def normalize(data):
    """Normalize data to [0, 1] range."""
    dmin, dmax = data.min(), data.max()
    if dmax - dmin < 1e-8:
        return np.zeros_like(data)
    return (data - dmin) / (dmax - dmin)


def save_slice_images(volume, output_prefix, label=""):
    """
    Save axial, coronal, sagittal slice images from a 3D volume.

    Args:
        volume: (H, W, D) numpy array
        output_prefix: output file path prefix
        label: text label for the image
    """
    norm = normalize(volume)
    h, w, d = norm.shape

    slices = {
        'axial': (norm[:, :, d // 2] * 255).astype(np.uint8),
        'coronal': (norm[:, w // 2, :] * 255).astype(np.uint8),
        'sagittal': (norm[h // 2, :, :] * 255).astype(np.uint8),
    }

    for view_name, slice_img in slices.items():
        img = Image.fromarray(slice_img)
        img.save(f"{output_prefix}_{view_name}.png")

    return slices


def save_comparison_image(real_slices, syn_slices, output_path):
    """
    Create side-by-side comparison image (real vs synthetic).

    Args:
        real_slices: dict of {view_name: numpy_array}
        syn_slices: dict of {view_name: numpy_array}
        output_path: output file path
    """
    views = ['axial', 'coronal', 'sagittal']
    images = []

    for view in views:
        real_img = Image.fromarray(real_slices[view]).convert('RGB')
        syn_img = Image.fromarray(syn_slices[view]).convert('RGB')

        # Resize to same height
        target_h = 200
        real_w = int(real_img.width * target_h / real_img.height)
        syn_w = int(syn_img.width * target_h / syn_img.height)
        real_img = real_img.resize((real_w, target_h))
        syn_img = syn_img.resize((syn_w, target_h))

        # Combine side by side with gap
        gap = 10
        combined_w = real_w + syn_w + gap
        combined = Image.new('RGB', (combined_w, target_h + 30), (0, 0, 0))
        combined.paste(real_img, (0, 30))
        combined.paste(syn_img, (real_w + gap, 30))

        # Add labels
        draw = ImageDraw.Draw(combined)
        draw.text((real_w // 2 - 20, 5), "Real", fill=(255, 255, 255))
        draw.text((real_w + gap + syn_w // 2 - 30, 5), "Synthetic", fill=(100, 255, 100))

        images.append(combined)

    # Stack views vertically
    total_h = sum(img.height for img in images) + 10 * (len(images) - 1)
    max_w = max(img.width for img in images)
    final = Image.new('RGB', (max_w, total_h), (0, 0, 0))

    y_offset = 0
    for img in images:
        final.paste(img, (0, y_offset))
        y_offset += img.height + 10

    final.save(output_path)
    LOG.info(f"Comparison image saved: {output_path}")


def extract_3d_mesh(volume, output_prefix, level=0.3):
    """
    Extract 3D mesh from volume using marching cubes.

    Args:
        volume: (H, W, D) numpy array
        output_prefix: output file path prefix
        level: isosurface threshold (0-1 range)

    Returns:
        (num_vertices, num_faces) or None if failed
    """
    norm = normalize(volume)

    try:
        verts, faces, normals, values = measure.marching_cubes(norm, level=level)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)

        mesh.export(f"{output_prefix}.obj")
        mesh.export(f"{output_prefix}.stl")

        LOG.info(f"3D mesh saved: {len(verts)} vertices, {len(faces)} faces")
        return len(verts), len(faces)

    except Exception as e:
        LOG.error(f"Mesh extraction failed: {e}")
        return None


def process_single_file(nii_path, output_dir, subject_id):
    """
    Process a single .nii.gz file: generate 2D slices and 3D meshes.

    Args:
        nii_path: path to .nii.gz file
        output_dir: output directory
        subject_id: subject identifier string
    """
    LOG.info(f"Processing {subject_id}...")

    img = nib.load(nii_path)
    data = img.get_fdata()
    LOG.info(f"  Shape: {data.shape}, Range: [{data.min():.4f}, {data.max():.4f}]")

    subject_dir = os.path.join(output_dir, subject_id)
    os.makedirs(subject_dir, exist_ok=True)

    # Handle different data shapes
    if len(data.shape) == 4 and data.shape[0] == 2:
        # (2, H, W, D) format: channel 0 = real, channel 1 = synthetic
        real_vol = data[0]
        syn_vol = data[1]

        LOG.info(f"  Real MRI range: [{real_vol.min():.4f}, {real_vol.max():.4f}]")
        LOG.info(f"  Synthetic MRI range: [{syn_vol.min():.4f}, {syn_vol.max():.4f}]")

        # 2D slices
        real_slices = save_slice_images(
            real_vol,
            os.path.join(subject_dir, "real"),
            label="Real"
        )
        syn_slices = save_slice_images(
            syn_vol,
            os.path.join(subject_dir, "synthetic"),
            label="Synthetic"
        )

        # Comparison image
        save_comparison_image(
            real_slices, syn_slices,
            os.path.join(subject_dir, "comparison.png")
        )

        # 3D meshes
        LOG.info("  Extracting real MRI 3D mesh...")
        extract_3d_mesh(
            real_vol,
            os.path.join(subject_dir, "real_brain"),
            level=0.3
        )

        LOG.info("  Extracting synthetic MRI 3D mesh...")
        extract_3d_mesh(
            syn_vol,
            os.path.join(subject_dir, "synthetic_brain"),
            level=0.3
        )

    elif len(data.shape) == 3:
        # (H, W, D) format: single volume
        save_slice_images(data, os.path.join(subject_dir, "volume"))
        extract_3d_mesh(data, os.path.join(subject_dir, "brain"), level=0.3)

    else:
        LOG.warning(f"  Unexpected shape: {data.shape}, skipping")
        return

    LOG.info(f"  [DONE] {subject_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Cor2Vox sampling results (2D slices + 3D meshes)"
    )
    parser.add_argument(
        '--input_dir', type=str, required=True,
        help='Directory containing .nii.gz sampling results'
    )
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='Output directory for visualizations'
    )
    parser.add_argument(
        '--level', type=float, default=0.3,
        help='Marching cubes threshold for 3D mesh extraction (default: 0.3)'
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Find all .nii.gz files
    nii_files = sorted([
        f for f in os.listdir(args.input_dir)
        if f.endswith('.nii.gz')
    ])

    LOG.info(f"Found {len(nii_files)} .nii.gz files in {args.input_dir}")

    for nii_file in nii_files:
        subject_id = nii_file.replace('.nii.gz', '').replace('_syn_mri', '')
        nii_path = os.path.join(args.input_dir, nii_file)
        process_single_file(nii_path, args.output_dir, subject_id)

    LOG.info(f"\n=== Visualization Complete ===")
    LOG.info(f"Output: {args.output_dir}")
    LOG.info(f"Processed: {len(nii_files)} files")


if __name__ == '__main__':
    main()
