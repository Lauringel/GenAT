from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

# import code
from ambrosio_tortorelli import SimpleMaskedATDetector
from checkpoints import (
    DEEPCRACK_FINETUNED,
    DEEPCRACK_PRETRAINED,
    VQGAN_FINETUNED_L1,
    VQGAN_FINETUNED_L2,
    get_deepcrack_checkpoint,
    get_vqgan_checkpoint,
)
from model_loading import load_dcmodel, load_vqgan_model_ckpt
from data_loading import image_transform, load_rgb_image, load_binary_mask
from postprocessing import Postprocessor, extract_tiles, stitch_soft_maps, stitch_binary_maps
from preprocessing import apply_preprocessing
from crackdetector import CrackDetector

# ===============================
# Switches
# ===============================

USE_PReg = True
VARIANT = "preg" if USE_PReg else "no_preg"
LR_Z = 0.005
LR_V = 0.1
ITERS_SYNTHETIC = 300
ITERS_MASKED_AT = 700
ITERS_REAL = 800
TILE_SIZE = 512
OVERLAP = 384

# ===============================
# Paths and device
# ===============================
PROJECT_ROOT = Path(__file__).resolve().parent
PICTURES_DIR = PROJECT_ROOT / "pictures"
CONFIG_DIR = PROJECT_ROOT / "configs"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
GRIDSEARCH_DIR = PROJECT_ROOT / "results" / "gridsearch"
OUT_DIR = PROJECT_ROOT / "results" / "plots_paper"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ===============================
# Column plot function for 
# different models and pictures
# ===============================

def plot_synthetic_examples(
    deepcrack_original,
    masked_at,
    masked_postprocessor,
    genat,
    genat_postprocessor,
):
    # painting patches and masks
    synthetic_pairs = [
        ("synthetic_01783.png", "mask_01783.png"),
        ("synthetic_01847.png", "mask_01847.png"),
        ("synthetic_01882.png", "mask_01882.png"),
    ]

    # titles for columns
    titles = [
        "Input",
        "Ground truth",
        "DeepCrack",
        "Masked AT",
        "GenAT",
    ]

    fig, axes = plt.subplots(
        nrows=len(synthetic_pairs),
        ncols=len(titles),
        figsize=(15, 9),
        squeeze=False,
    )

    for row, (image_name, mask_name) in enumerate(synthetic_pairs):
        image_pil = load_rgb_image(PICTURES_DIR / image_name)
        gt = load_binary_mask(PICTURES_DIR / mask_name)

        image = image_transform(image_pil).unsqueeze(0).to(DEVICE)

        # Original DeepCrack uses BGR input.
        with torch.no_grad():
            output = deepcrack_original(image[:, [2, 1, 0], :, :])

            if isinstance(output, (list, tuple)):
                output = output[0]

            deepcrack_binary = (torch.sigmoid(output) > 0.5).float()[0, 0].cpu().numpy()

        with torch.enable_grad():
            _, v_masked = masked_at(image, ITERS_MASKED_AT)

        _, masked_binary = masked_postprocessor(v_masked)

        with torch.enable_grad():
            _, v_genat, _ = genat(image)

        _, genat_binary = genat_postprocessor(v_genat)

        image_np = (
            image[0]
            .detach()
            .cpu()
            .permute(1, 2, 0)
            .numpy()
        )

        results = [
            image_np,
            gt,
            deepcrack_binary,
            masked_binary,
            genat_binary,
        ]

        for column, result in enumerate(results):
            if column == 0:
                axes[row, column].imshow(result)
            else:
                axes[row, column].imshow(result, cmap="gray", vmin=0, vmax=1)
            axes[row, column].axis("off")

            if row == 0:
                axes[row, column].set_title(titles[column])

    fig.tight_layout()
    output_path = OUT_DIR / f"synthetic_model_comparison_{VARIANT}.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight",)
    plt.close(fig)

    print(f"Saved synthetic comparison: {output_path}", flush=True)


# ===============================
# Single image plot function
# ===============================

def plot_real_image(
    picture_name,
    output_stem,
    vqgan,
    deepcrack,
    mu,
    eps,
    lam_AT,
    lambda_dc,
    clip_limit,
    kernel_size,
    sigma_color,
    sigma_spatial,
):
    image_pil = load_rgb_image(PICTURES_DIR / picture_name)
    img_np_orig = np.asarray(image_pil)

    img_np = apply_preprocessing(
        img_np_orig,
        mode="lab_clahe",
        clip_limit=clip_limit,
        kernel_size=kernel_size,
        sigma_color=sigma_color,
        sigma_spatial=sigma_spatial,
    )

    tiles, h_orig, w_orig, h_pad, w_pad = extract_tiles(
        img_np,
        tile_size=TILE_SIZE,
        overlap=OVERLAP,
    )

    detector = CrackDetector(
        vqgan=vqgan,
        dcmodel=deepcrack if lambda_dc != 0.0 else None,
        epsi=float(eps),
        lam_preg=float(lam_AT) if USE_PReg else 0.0,
        lam_creg=float(lam_AT) * float(mu),
        lam_crack=float(lambda_dc),
        iters=ITERS_REAL,
        lr_z=LR_Z,
        lr_v=LR_V,
    )

    postprocessor = Postprocessor("otsu")
    soft_tiles = []
    binary_tiles = []

    for x, y, tile_img in tiles:
        tile_tensor = image_transform(
            Image.fromarray(tile_img)
        ).unsqueeze(0).to(DEVICE)

        with torch.enable_grad():
            _, v, _ = detector(tile_tensor)

        crack_soft, crack_binary = postprocessor(v)

        soft_tiles.append((x, y, crack_soft))
        binary_tiles.append((x, y, crack_binary))

    # Stitch maps
    crack_soft_full = stitch_soft_maps(
        soft_tiles,
        h_pad,
        w_pad,
        tile_size=TILE_SIZE,
    )[:h_orig, :w_orig]

    crack_binary_full = stitch_binary_maps(
        binary_tiles,
        h_pad,
        w_pad,
        tile_size=TILE_SIZE,
    )[:h_orig, :w_orig]

    preprocessed_path = OUT_DIR / (
        f"{output_stem}_preprocessed.png"
    )
    soft_path = OUT_DIR / f"{output_stem}_crack_soft.png"
    binary_path = OUT_DIR / f"{output_stem}_crack_binary.png"
    overlay_path = OUT_DIR / f"{output_stem}_crack_overlay.png"

    Image.fromarray(img_np).save(preprocessed_path)

    # Save soft crack map (grayscale)
    soft_uint8 = (
        crack_soft_full * 255
    ).clip(0, 255).astype(np.uint8)

    # Save binary crack map
    binary_uint8 = (
        crack_binary_full * 255
    ).astype(np.uint8)

    Image.fromarray(soft_uint8, mode="L").save(soft_path)
    Image.fromarray(binary_uint8, mode="L").save(binary_path)

    # Convert original image to float in [0,1]
    img_float = img_np_orig.astype(np.float32) / 255.0

    mask = crack_binary_full[..., None]

    # overlay color
    overlay_color = np.array([0.0, 1.0, 1.0], dtype=np.float32) # cyan, for magenta use [1.0, 0.0, 1.0]

    # transparency
    alpha = 0.7

    # create overlay
    overlay = (
        img_float * (1.0 - alpha * mask)
        + overlay_color * alpha * mask
    )
    
    # Convert to uint8 for saving
    overlay_uint8 = (
        np.clip(overlay, 0.0, 1.0) * 255
    ).astype(np.uint8)

    # Save overlay image
    Image.fromarray(overlay_uint8).save(overlay_path)

    print(
        f"Saved {output_stem} results:\n"
        f"{preprocessed_path}\n"
        f"{soft_path}\n"
        f"{binary_path}\n"
        f"{overlay_path}",
        flush=True,
    )

    return crack_soft_full, crack_binary_full


def plot_brugghen(vqgan, deepcrack, best):
    return plot_real_image(
        picture_name="Brugghen_patch.jpg",
        output_stem=f"Brugghen_Zecher_{VARIANT}",
        vqgan=vqgan,
        deepcrack=deepcrack,
        mu=0.5 * float(best["mu"]),
        eps=0.25 * float(best["eps"]),
        lam_AT=0.4 * float(best["lam"]),
        lambda_dc=0.1,
        clip_limit=0.006,
        kernel_size=80,
        sigma_color=0.01,
        sigma_spatial=1,
    )


def plot_lorrain(vqgan, deepcrack, best, lambda_dc):
    return plot_real_image(
        picture_name="Lorrain_patch.jpg",
        output_stem=f"Lorrain_Wueste_{VARIANT}",
        vqgan=vqgan,
        deepcrack=deepcrack,
        mu=2.0 * float(best["mu"]),
        eps=float(best["eps"]),
        lam_AT=float(best["lam"]),
        lambda_dc=float(lambda_dc),
        clip_limit=0.01,
        kernel_size=128,
        sigma_color=0.015,
        sigma_spatial=2,
    )


# ==================================================
# MAIN:
# - Load best csv rows.
# - Resolve checkpoints through checkpoints.py.
# - Load VQGAN and both required DeepCrack versions.
# - Call plot_synthetic_examples(...).
# - Call plot_brugghen_zecher(...).
# - Call plot_lorrain_wueste(...).
# ==================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------
    # Read grid-search results
    # ---------------------------------

    genat_variant = ("preg" if USE_PReg else "no_preg")

    genat_csv = GRIDSEARCH_DIR / (f"validation_{genat_variant}_gridsearch_results.csv")
    masked_at_csv = GRIDSEARCH_DIR / ("validation_maskedAT_gridsearch_results.csv")
    lambda_dc_csv = GRIDSEARCH_DIR / (f"validation_{genat_variant}_lambda_dc_results.csv")

    genat_df = pd.read_csv(genat_csv)
    genat_best = genat_df.loc[genat_df["F1"].idxmax()]

    masked_at_df = pd.read_csv(masked_at_csv)
    masked_at_best = masked_at_df.loc[masked_at_df["F1"].idxmax()]

    lambda_dc_df = pd.read_csv(lambda_dc_csv)
    lambda_dc_best = lambda_dc_df.loc[lambda_dc_df["F1"].idxmax()]
    lambda_dc = float(lambda_dc_best["lambda_dc"])

    print(
        "Best GenAT parameters:\n"
        f"mu = {genat_best['mu']}\n"
        f"eps = {genat_best['eps']}\n"
        f"lam = {genat_best['lam']}\n"
        f"threshold = {genat_best['threshold']}\n"
        f"lambda_dc = {lambda_dc}",
        flush=True,
    )

    # ---------------------------------
    # Resolve and load checkpoints
    # ---------------------------------

    VQGAN_CONFIG = CONFIG_DIR / "vqgan_config.yaml"
    
    vqgan_path = get_vqgan_checkpoint((VQGAN_FINETUNED_L1 if USE_PReg else VQGAN_FINETUNED_L2), cache_dir=CHECKPOINT_DIR)

    original_dc_path = get_deepcrack_checkpoint(DEEPCRACK_PRETRAINED, cache_dir=CHECKPOINT_DIR)

    finetuned_dc_path = get_deepcrack_checkpoint(DEEPCRACK_FINETUNED, cache_dir=CHECKPOINT_DIR)

    vqgan = load_vqgan_model_ckpt(VQGAN_CONFIG, vqgan_path, DEVICE)
      
    for parameter in vqgan.parameters():
        parameter.requires_grad = False

    deepcrack_original = load_dcmodel(
        original_dc_path,
        DEVICE,
        is_checkpoint=False,
    )

    deepcrack_finetuned = load_dcmodel(
        finetuned_dc_path,
        DEVICE,
        is_checkpoint=True,
    )

    # ---------------------------------
    # Construct synthetic-image models
    # ---------------------------------

    masked_at = SimpleMaskedATDetector(
        mu=float(masked_at_best["mu"]),
        eps=float(masked_at_best["eps"]),
        lam_AT=float(masked_at_best["lam"])
    ).to(DEVICE)

    masked_postprocessor = Postprocessor(masked_at_best["threshold"])

    genat = CrackDetector(
        vqgan=vqgan,
        dcmodel=(
            deepcrack_finetuned
            if lambda_dc != 0.0
            else None
        ),
        epsi=float(genat_best["eps"]),
        lam_preg=(
            float(genat_best["lam"])
            if USE_PReg
            else 0.0
        ),
        lam_creg=(
            float(genat_best["lam"])
            * float(genat_best["mu"])
        ),
        lam_crack=lambda_dc,
        iters=ITERS_SYNTHETIC,
        lr_z=LR_Z,
        lr_v=LR_V,
    )

    genat_postprocessor = Postprocessor(genat_best["threshold"])

    # ---------------------------------
    # Create paper figures
    # ---------------------------------

    plot_synthetic_examples(
        deepcrack_original=deepcrack_original,
        masked_at=masked_at,
        masked_postprocessor=masked_postprocessor,
        genat=genat,
        genat_postprocessor=genat_postprocessor,
    )

    plot_brugghen(vqgan=vqgan, deepcrack=deepcrack_finetuned, best=genat_best)

    plot_lorrain(vqgan=vqgan, deepcrack=deepcrack_finetuned, best=genat_best, lambda_dc=lambda_dc)
    

if __name__ == "__main__":
    main()
