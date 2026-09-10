"""Image preprocessing (contrast enhancement) applied before running the crack detector."""

import numpy as np
from skimage import exposure, color
from skimage.restoration import denoise_bilateral


def preprocess_lab_clahe(img_np, clip_limit=0.01, kernel_size=128, sigma_color=0.015, sigma_spatial=2):
    img_float = img_np.astype(np.float32) / 255.0
    lab = color.rgb2lab(img_float)
    L = lab[:, :, 0] / 100.0
    
    # kernel_size controls the size of the local regions (tiles) CLAHE operates on.
    L_eq = exposure.equalize_adapthist(L, clip_limit=clip_limit, kernel_size=kernel_size)

    # optional mild edge-preserving denoising
    L_eq = denoise_bilateral(
        L_eq,
        sigma_color=sigma_color,  # Zecher 0.01
        sigma_spatial=sigma_spatial,    # Zecher 1
        channel_axis=None
    )
    lab[:, :, 0] = L_eq * 100.0
    rgb_eq = color.lab2rgb(lab)

    return (rgb_eq * 255).clip(0, 255).astype(np.uint8)


def apply_preprocessing(img_np, mode, clip_limit=0.01, kernel_size=128, sigma_color=0.015, sigma_spatial=2):
    if mode == "none":
        return img_np
    elif mode == "lab_clahe":
        return preprocess_lab_clahe(
            img_np,
            clip_limit=clip_limit,
            kernel_size=kernel_size,
            sigma_color=sigma_color,
            sigma_spatial=sigma_spatial,
        )
    else:
        raise ValueError(f"Unknown preprocess_mode: {mode}")
