"""
Gridsearch for hyperparameter selection for Amrosio-Tortorelli and GenAT.

Stage 1 selects μ, ε, λ_AT, and the post-processing threshold on a validation split.
Lists for mu_list, eps_list, lam_list are created.
For each triple, CrackDetector(vqgan=vq, mu=mu, eps=eps, lam_AT=lam) is instantiated and evaluation is done on few images from val_loader with short runs.

For GenAT, stage 2 keeps those values fixed and selects the
DeepCrack prior weight "lambda_dc" on the same validation split.
"""

# ===========================================================================================
# Unified Grid Search: Simple AT / Simple Masked AT / GenAT (R_{PReg}) with/without DeepCrack
# ===========================================================================================

import argparse
import os
import glob
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image, ImageFile
from itertools import product

ImageFile.LOAD_TRUNCATED_IMAGES = True

# code import
from model_loading import load_vqgan_model_ckpt, load_dcmodel
from data_loading import get_val_loader
from metrics import compute_metrics
from math_utils import spatial_grad
from crackdetector import CrackDetector
from checkpoints import (
    get_vqgan_checkpoint,
    get_deepcrack_checkpoint,
    VQGAN_FINETUNED_L1,
    VQGAN_FINETUNED_L2
)
from ambrosio_tortorelli import ambrosio_tortorelli_rgb, SimpleMaskedATDetector, SimpleATDetector
from postprocessing import normalize_crack_map, custom_threshold

# ============================================================
# Paths / Device
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCRIPT_DIR = Path(__file__).resolve().parent

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-images", type=Path, required=True)
    parser.add_argument("--val-masks", type=Path, required=True)

    parser.add_argument("--vqgan-config", type=Path, default=SCRIPT_DIR / "configs" / "vqgan_config.yaml")
    parser.add_argument("--vqgan-pretrained", type=Path, default=None)
    parser.add_argument("--vqgan-finetuned", type=Path, default=None)
    parser.add_argument("--deepcrack-checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=SCRIPT_DIR / "checkpoints")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "results" / "gridsearch")
    parser.add_argument("--device", default=None)

    return parser.parse_args()

# ============================================================
# Switches
# ============================================================

USE_VQGAN = True
USE_FINETUNED = True # use fine-tuned version of VQGAN?
SimpleMaskedAT = False # used only if USE_VQGAN = False
USE_PReg = True # True for using mixed gradient term R_{PReg}, 0 for version without R_{PReg}

# ============================================================
# Grid
# ============================================================

mu_list  = [0.01, 0.1, 0.5, 2] # the higher mu, the smoother v
eps_list = [0.005, 0.01, 0.05, 0.1] # controls widths of detected edges (lower eps, thinner lines)
lam_list = [0.1, 0.5, 1.0, 5.0]
threshold_grid = [97, 99, "otsu"]
AT_GRID_MAX_IMAGES = 5
lambda_dc_list = [0.0, 0.2, 0.5, 0.7, 1.0]

ITERS_GRID     = 120   # grid search / quick eval
ITERS_LAMBDA_DC = 300  # second-stage crack-prior-weight search
ITERS_AT       = 700   # simple (Masked) Ambrosio-Tortorelli

# ============================================================
# Optimization parameters
# ============================================================

lr_v = 0.1
lr_z = 0.005
dt = 0.2 # step size for simple AT
BATCH_SIZE = 1

# ============================================================
# Factory
# ============================================================
def make_detector(vq, dcmodel, params, device, iters, lam_crack=0.0):
    mu = params["mu"]
    eps = params["eps"]
    lam = params["lam"]

    if USE_VQGAN:
        return CrackDetector(
            vqgan=vq,
            dcmodel=dcmodel,
            epsi=eps,
            lam_preg=lam if USE_PReg else 0.0,
            lam_creg=lam * mu,
            lam_crack=lam_crack,
            iters=iters,
            lr_z=lr_z,
            lr_v=lr_v,
        )

    if SimpleMaskedAT:
        return SimpleMaskedATDetector(
            mu=mu,
            eps=eps,
            lam_AT=lam,
        ).to(device)

    return SimpleATDetector(
        mu=mu,
        eps=eps,
        lam_AT=lam,
        dt=dt,
    ).to(device)


def run_detector(detector, img, iters):
    if isinstance(detector, CrackDetector):
        return detector(img)

    Gz, v = detector(img, iters)
    return Gz, v, None


# ============================================================
# Evaluation
# ============================================================
def quick_eval(detector, val_loader, device, iters, threshold_modes=None, max_images=None):
    if threshold_modes is None:
        threshold_modes = threshold_grid
    
    scores = {
        mode: {
            "F1": [],
            "IoU": [],
            "precision": [],
            "recall": []
        }
        for mode in threshold_modes
    }

    for i, (img, gt, _) in enumerate(val_loader):
        if max_images is not None and i >= max_images:
            break

        img = img.to(device)
        gt = gt.to(device)

        Gz, v, crack_prior = run_detector(detector, img, iters)

        crack_map_norm = normalize_crack_map(v)

        for mode in threshold_modes:
            crack_mask = custom_threshold(crack_map_norm, mode)

            F1, IoU, precision, recall, _ = compute_metrics(crack_mask, gt.squeeze())

            scores[mode]["F1"].append(F1)
            scores[mode]["IoU"].append(IoU)
            scores[mode]["precision"].append(precision)
            scores[mode]["recall"].append(recall)

        del Gz, v, crack_prior

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        mode: {
            metric: float(np.mean(values))
            for metric, values in mode_scores.items()
        }
        for mode, mode_scores in scores.items()
    }


def grid_search_AT(vq, dcmodel, val_loader, device, iters):
    results = []

    for mu, eps, lam in product(mu_list, eps_list, lam_list):
        params = {
            "mu": mu,
            "eps": eps,
            "lam": lam
        }

        print(
            f"\n Testing mu={mu}, eps={eps}, lam={lam}",
            flush=True
        )

        detector = make_detector(
            vq,
            dcmodel,
            params,
            device,
            iters
        )

        threshold_scores = quick_eval(
            detector,
            val_loader,
            device,
            iters=iters,
            max_images=AT_GRID_MAX_IMAGES
        )

        for threshold_mode, metrics in threshold_scores.items():
            results.append({
                "mu": mu,
                "eps": eps,
                "lam": lam,
                "threshold": threshold_mode,
                **metrics,
            })

        del detector
        
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return results


def grid_search_lambda_dc(vq, dcmodel, val_loader, device, best_base, iters):
    """Select lambda_dc while keeping the best stage-1 settings fixed."""
    params = {
        "mu": float(best_base["mu"]),
        "eps": float(best_base["eps"]),
        "lam": float(best_base["lam"]),
    }
    threshold_mode = best_base["threshold"]
    results = []

    for lambda_dc in lambda_dc_list:
        print(
            f"\n Testing lambda_dc={lambda_dc} with fixed "
            f"mu={params['mu']}, eps={params['eps']}, "
            f"lam={params['lam']}, threshold={threshold_mode}",
            flush=True,
        )

        detector = make_detector(
            vq=vq,
            dcmodel=dcmodel,
            params=params,
            device=device,
            iters=iters,
            lam_crack=float(lambda_dc),
        )

        threshold_scores = quick_eval(
            detector=detector,
            val_loader=val_loader,
            device=device,
            iters=iters,
            threshold_modes=[threshold_mode],
            max_images=None,
        )

        results.append({
            **params,
            "threshold": threshold_mode,
            "lambda_dc": float(lambda_dc),
            **threshold_scores[threshold_mode],
        })

        del detector

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return results


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    args = parse_args()

    if USE_VQGAN and args.vqgan_pretrained is None:
        raise ValueError(
            "--vqgan-pretrained is required when USE_VQGAN=True."
        )
    
    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    
    val_loader = get_val_loader(
        image_dir=args.val_images,
        mask_dir=args.val_masks,
        batch_size=1,
    )

    # stage 1
    if USE_VQGAN:
        if USE_FINETUNED:
            print("🔄 Loading finetuned VQGAN...", flush=True)
            
            vqgan_checkpoint = (
                args.vqgan_finetuned
                if args.vqgan_finetuned is not None
                else get_vqgan_checkpoint(
                    VQGAN_FINETUNED_L1 if USE_PReg else VQGAN_FINETUNED_L2,
                    cache_dir=args.checkpoint_dir,
                )
            )
        else:
            if args.vqgan_pretrained is None:
                raise ValueError("--vqgan-pretrained is required when USE_FINETUNED=False.")

            print("🔄 Loading original pretrained VQGAN...", flush=True)
            vqgan_checkpoint = args.vqgan_pretrained

        vq = load_vqgan_model_ckpt(
            args.vqgan_config,
            vqgan_checkpoint,
            device
        )

        for p in vq.parameters():
            p.requires_grad = False
            
        print("✅ Loaded finetuned VQGAN.", flush=True)
            
        deepcrack_checkpoint = (
            args.deepcrack_checkpoint
            if args.deepcrack_checkpoint is not None
            else get_deepcrack_checkpoint(
                cache_dir=args.checkpoint_dir,
            )
        )

        dcmodel = load_dcmodel(
            deepcrack_checkpoint,
            device,
            is_checkpoint=True,
        )

        iters = ITERS_GRID

    else:
        vq = None
        dcmodel = None
        iters = ITERS_AT

    if device.type == "cuda":
        torch.cuda.empty_cache()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if USE_VQGAN:
        variant = "preg" if USE_PReg else "no_preg"
    elif SimpleMaskedAT:
        variant = "maskedAT"
    else:
        variant = "simpleAT"
    csv_path = (args.output_dir / f"validation_{variant}_gridsearch_results.csv")
    
    results = grid_search_AT(
        vq=vq,
        dcmodel=dcmodel,
        val_loader=val_loader,
        device=device,
        iters=iters
    )

    df = pd.DataFrame(results)
    df["use_preg"] = USE_PReg
    df["lam_crack"] = 0.0
    df["lr_v"] = lr_v
    df["lr_z"] = lr_z
    df["iters"] = iters
    df.to_csv(csv_path, index=False)
    print("\n🏆 Best by F1:", df.loc[df["F1"].idxmax()].to_dict())
    print("\n🏆 Best by IoU:", df.loc[df["IoU"].idxmax()].to_dict())
    print(f"✅ Saved grid-search results to: {csv_path}", flush=True)

    # stage 2
    if USE_VQGAN:
        best_base = df.loc[df["F1"].idxmax()]
        lambda_dc_results = grid_search_lambda_dc(
            vq=vq,
            dcmodel=dcmodel,
            val_loader=val_loader,
            device=device,
            best_base=best_base,
            iters=ITERS_LAMBDA_DC
        )

        lambda_dc_df = pd.DataFrame(lambda_dc_results)
        lambda_dc_df["use_preg"] = USE_PReg
        lambda_dc_df["lr_v"] = lr_v
        lambda_dc_df["lr_z"] = lr_z
        lambda_dc_df["iters"] = ITERS_LAMBDA_DC

        lambda_dc_path = (args.output_dir / f"validation_{variant}_lambda_dc_results.csv")
        lambda_dc_df.to_csv(lambda_dc_path, index=False)

        print("\n🏆 Best lambda_dc by F1:", lambda_dc_df.loc[lambda_dc_df["F1"].idxmax()].to_dict())
        print(f"✅ Saved lambda_dc results to: {lambda_dc_path}", flush=True)
