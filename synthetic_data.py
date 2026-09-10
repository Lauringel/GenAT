"""
Create a single synthetic cracked painting by overlaying a given crack mask
onto a given crack-free painting image.

The crack mask is assumed to be binary (0/255) and is overlaid while preserving
the crack geometry.
"""

import argparse
import os

import cv2
import numpy as np

TARGET_SIZE = (512, 512)


def load_painting(path):
    painting = cv2.imread(path)
    if painting is None:
        raise RuntimeError(f"Could not read painting image: {path}")
    return cv2.resize(painting, TARGET_SIZE, interpolation=cv2.INTER_AREA)


def load_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Could not read mask image: {path}")
    mask = cv2.resize(mask, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)

    mask_float = (mask > 0).astype(np.float32)
    if mask_float.max() == 0:
        raise RuntimeError("Mask is completely black (no crack pixels).")
    return mask_float


def overlay_cracks(painting, mask_float):
    # Overlay dark cracks onto bright paintings and vice versa, so the cracks stay visible.
    brightness = painting.mean() / 255.0
    crack_type = "dark" if brightness > 0.5 else "bright"

    alpha = np.random.uniform(0.5, 0.8)  # blending strength
    painting_f = painting.astype(np.float32)

    if crack_type == "dark":
        overlay = painting_f * (1.0 - alpha * mask_float[..., None])
    else:
        overlay = painting_f + alpha * 255.0 * mask_float[..., None]

    return np.clip(overlay, 0, 255).astype(np.uint8)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--painting", required=True, help="Path to the crack-free painting image.")
    parser.add_argument("--mask", required=True, help="Path to the binary crack mask (0/255).")
    parser.add_argument("--out-dir", required=True, help="Directory to save the synthetic image and mask to.")
    return parser.parse_args()


def main():
    args = parse_args()

    painting = load_painting(args.painting)
    mask_float = load_mask(args.mask)
    synthetic_img = overlay_cracks(painting, mask_float)

    os.makedirs(args.out_dir, exist_ok=True)
    img_out = os.path.join(args.out_dir, "synthetic_single.png")
    mask_out = os.path.join(args.out_dir, "mask_single.png")

    cv2.imwrite(img_out, synthetic_img)
    cv2.imwrite(mask_out, (mask_float * 255).astype(np.uint8))

    print("Single synthetic cracked image created.")
    print(f"   Image saved to: {img_out}")
    print(f"   Mask saved to:  {mask_out}")


if __name__ == "__main__":
    main()