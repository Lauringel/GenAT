"""
Take all models, that is 
- the original DeepCrack model,
- the fine-tuned DeepCrack model,
- the simple Ambrosio-Tortorelli approach,
- the proposed GenAT R_{PReg} approach including the mixed gradient term R_{PReg} (corresponding to lambda_preg > 0), 
- and GenAT without the mixed gradient term R_{PReg} (corresponding to lambda_preg = 0)
and for the best evaluated values for mu, lambda and epsilon evaluate models (F1) on the (synthtic) testset in order to compare their performance.
"""

import argparse
from pathlib import Path
import torch
import pandas as pd
import matplotlib.pyplot as plt

# import code
from ambrosio_tortorelli import SimpleMaskedATDetector
from checkpoints import (
    VQGAN_FINETUNED_L1,
    VQGAN_FINETUNED_L2,
    get_deepcrack_checkpoint,
    get_vqgan_checkpoint,
)
from crackdetector import CrackDetector
from data_loading import get_test_loader
from metrics import compute_metrics
from model_loading import load_dcmodel, load_vqgan_model_ckpt
from postprocessing import Postprocessor


PROJECT_ROOT = Path(__file__).resolve().parent

# =========================
# Paths
# =========================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the crack-detection models on the test set."
    )

    parser.add_argument("--test-images", type=Path, required=True)
    parser.add_argument("--test-masks", type=Path, required=True)

    parser.add_argument(
        "--gridsearch-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "gridsearch",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "evaluation",
    )
    parser.add_argument(
        "--vqgan-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "vqgan_config.yaml",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "checkpoints",
    )
    parser.add_argument(
        "--deepcrack-checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--vqgan-l1-checkpoint",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--vqgan-l2-checkpoint",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--deepcrack-finetuned-checkpoint",
        type=Path,
        default=None,
    )

    return parser.parse_args()

args = parse_args()

# =========================
# GLOBALS (same as training)
# =========================
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

ITERS_TESTSET = 300
ITERS_AT = 700
LR_Z = 0.005
LR_V = 0.1

METRIC_COLUMNS = ["F1", "IoU", "Precision", "Recall", "Accuracy"]

# =========================
# MODEL DEFINITIONS
# =========================
models = [
    {
        "name": "DeepCrack",
        "kind": "deepcrack",
        "dc_ckpt": args.deepcrack_checkpoint,
        "is_checkpoint": False,
        "finetuned": False,
    },
    {
        "name": "Fine-tuned DeepCrack",
        "kind": "deepcrack",
        "dc_ckpt": args.deepcrack_finetuned_checkpoint,
        "is_checkpoint": True,
        "finetuned": True,
    },
    {
        "name": "Masked AT",
        "kind": "masked_at",
        "csv": args.gridsearch_dir / "validation_maskedAT_gridsearch_results.csv",
    },
    {
        "name": r"GenAT Reg_{PReg}",
        "kind": "genat",
        "use_preg": True,
        "csv": args.gridsearch_dir / "validation_preg_gridsearch_results.csv",
        "csv_dc": args.gridsearch_dir / "validation_preg_lambda_dc_results.csv",
        "vqgan_checkpoint_id": VQGAN_FINETUNED_L1,
        "dc_ckpt": args.deepcrack_finetuned_checkpoint,
        "vqgan_checkpoint_override": args.vqgan_l1_checkpoint
    },
    {
        "name": "GenAT",
        "kind": "genat",
        "use_preg": False,
        "csv": args.gridsearch_dir / "validation_no_preg_gridsearch_results.csv",
        "csv_dc": args.gridsearch_dir / "validation_no_preg_lambda_dc_results.csv",
        "vqgan_checkpoint_id": VQGAN_FINETUNED_L2,
        "dc_ckpt": args.deepcrack_finetuned_checkpoint,
        "vqgan_checkpoint_override": args.vqgan_l2_checkpoint
    },
]
print("Loaded Models.", flush=True)

# =========================
# Helpers
# =========================

def best_by_f1(csv_path):
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Validation CSV not found: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    if "F1" not in df.columns:
        raise ValueError(f"No F1 column in {csv_path}")

    scores = pd.to_numeric(df["F1"], errors="coerce")

    if scores.isna().all():
        raise ValueError(f"No valid F1 values in {csv_path}")

    return df.loc[scores.idxmax()]


def resolve_vqgan_checkpoint(model, args):
    override = model.get("vqgan_checkpoint_override")

    if override is not None:
        return override

    return get_vqgan_checkpoint(
        model["vqgan_checkpoint_id"],
        cache_dir=args.checkpoint_dir,
    )


def resolve_finetuned_deepcrack(model, args):
    if model.get("dc_ckpt") is not None:
        return model["dc_ckpt"]

    return get_deepcrack_checkpoint(
        cache_dir=args.checkpoint_dir,
    )


def get_lambda_dc(best_dc):
    if "lambda_dc" in best_dc.index:
        return float(best_dc["lambda_dc"])

    if "lam_crack" in best_dc.index:
        return float(best_dc["lam_crack"])

    raise ValueError(
        "Lambda-DC CSV needs a 'lambda_dc' or 'lam_crack' column."
    )


def scalar(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())

    return float(value)


def cached_metrics_complete(csv_path, expected_images):
    if not csv_path.exists():
        return False

    df = pd.read_csv(csv_path)
    required_columns = ["ImageIndex", *METRIC_COLUMNS]

    return (
        all(column in df.columns for column in required_columns)
        and len(df) == expected_images
    )


def evaluate_model(model, test_loader, args):
    metrics = {
        column: []
        for column in METRIC_COLUMNS
    }

    kind = model["kind"]

    if kind == "deepcrack":
        checkpoint = model["dc_ckpt"]

        if checkpoint is None:
            if model["finetuned"]:
                checkpoint = resolve_finetuned_deepcrack(
                    model,
                    args,
                )
            else:
                raise ValueError(
                    "--deepcrack-checkpoint is required when "
                    "the DeepCrack cache does not exist."
                )

        detector = load_dcmodel(
            checkpoint,
            DEVICE,
            is_checkpoint=model["is_checkpoint"],
        )
        detector.eval()

    elif kind == "masked_at":
        best = best_by_f1(model["csv"])

        detector = SimpleMaskedATDetector(
            mu=float(best["mu"]),
            eps=float(best["eps"]),
            lam_AT=float(best["lam"]),
        ).to(DEVICE)
        detector.eval()

        postprocessor = Postprocessor(best["threshold"])

    elif kind == "genat":
        best = best_by_f1(model["csv"])
        best_dc = best_by_f1(model["csv_dc"])

        vqgan_checkpoint = resolve_vqgan_checkpoint(
            model,
            args,
        )
        vq = load_vqgan_model_ckpt(
            args.vqgan_config,
            vqgan_checkpoint,
            DEVICE,
        )
        vq.eval()

        for parameter in vq.parameters():
            parameter.requires_grad = False

        deepcrack_checkpoint = resolve_finetuned_deepcrack(
            model,
            args,
        )
        deepcrack = load_dcmodel(
            deepcrack_checkpoint,
            DEVICE,
            is_checkpoint=True,
        )
        deepcrack.eval()

        for parameter in deepcrack.parameters():
            parameter.requires_grad = False

        lam_at = float(best["lam"])

        detector = CrackDetector(
            vqgan=vq,
            dcmodel=deepcrack,
            epsi=float(best["eps"]),
            lam_preg=lam_at if model["use_preg"] else 0.0,
            lam_creg=lam_at * float(best["mu"]),
            lam_crack=get_lambda_dc(best_dc),
            iters=ITERS_TESTSET,
            lr_z=LR_Z,
            lr_v=LR_V,
        )

        postprocessor = Postprocessor(best["threshold"])

    else:
        raise ValueError(f"Unknown model kind: {kind}")

    for image_index, (img, gt, _) in enumerate(test_loader):
        img = img.to(DEVICE)
        gt = gt.to(DEVICE)

        if kind == "deepcrack":
            with torch.no_grad():
                if model["finetuned"]:
                    mean = img.new_tensor(
                        [0.485, 0.456, 0.406]
                    ).view(1, 3, 1, 1)
                    std = img.new_tensor(
                        [0.229, 0.224, 0.225]
                    ).view(1, 3, 1, 1)

                    output = detector((img - mean) / std)

                    if isinstance(output, (list, tuple)):
                        output = output[-1]

                else:
                    # Preserve original DeepCrack BGR input.
                    output = detector(
                        img[:, [2, 1, 0], :, :]
                    )

                    if isinstance(output, (list, tuple)):
                        output = output[0]

                probability = torch.sigmoid(output)
                crack_bin = (
                    probability > 0.5
                ).float()[0, 0]

        elif kind == "masked_at":
            _, v = detector(img, ITERS_AT)
            _, crack_bin_raw = postprocessor(v)

            crack_bin = torch.as_tensor(
                crack_bin_raw,
                dtype=gt.dtype,
                device=gt.device,
            ).squeeze()

        else:
            _, v, _ = detector(img)
            _, crack_bin_raw = postprocessor(v)

            crack_bin = torch.as_tensor(
                crack_bin_raw,
                dtype=gt.dtype,
                device=gt.device,
            ).squeeze()

        values = compute_metrics(
            crack_bin,
            gt[0, 0],
        )

        for column, value in zip(METRIC_COLUMNS, values):
            metrics[column].append(scalar(value))

    per_image_df = pd.DataFrame({
        "ImageIndex": range(len(metrics["F1"])),
        **metrics,
    })

    del detector

    if kind == "genat":
        del vq
        del deepcrack

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return per_image_df

# =========================
# Main: Evaluation
# =========================
    
def main():
    args.output_dir.mkdir(parents=True, exist_ok=True)

    test_loader = get_test_loader(
        image_dir=args.test_images,
        mask_dir=args.test_masks,
        batch_size=1,
    )

    results = {}
    per_image_results = {}
    print("Starting evaluation ...", flush=True)
    
    for model in models:
        print(f'Evaluating model {model["name"]}', flush=True)
    
        safe_name = model["name"].replace(" ", "_").replace("/", "_").replace("{", "").replace("}", "")
        per_image_path = args.output_dir / f"test_metrics_{safe_name}.csv"
    
        
        if cached_metrics_complete(
            per_image_path,
            expected_images=len(test_loader.dataset),
        ):
            print(f"📂 Loading cached metrics: {per_image_path}", flush=True)
            per_image_df = pd.read_csv(per_image_path)
        else:
            print(f'Evaluating {model["name"]} ...', flush=True)
            per_image_df = evaluate_model(
                model=model,
                test_loader=test_loader,
                args=args,
            )
            per_image_df.to_csv(per_image_path, index=False)
            print(f"Saved metrics: {per_image_path}", flush=True)
    
        per_image_results[model["name"]] = per_image_df

        results[model["name"]] = {
            column: float(per_image_df[column].mean())
            for column in METRIC_COLUMNS
        }
    
    summary_df = pd.DataFrame([
        {"Model": model_name, **values}
        for model_name, values in results.items()
    ])

    summary_path = (
        args.output_dir
        / "test_metrics_summary.csv"
    )
    summary_df.to_csv(summary_path, index=False)
    print(f"💾 Saved summary metrics csv to {summary_path}", flush=True)
    
    # =========================
    # Plotting
    # =========================
    
    # Bar plot of MEAN F1 for all models
    plt.figure(figsize=(6,4))
    plt.bar(results.keys(), [values["F1"] for values in results.values()])
    plt.ylabel("Mean F1 score")
    plt.title("Best-validation-parameter performance on the test set")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    mean_plot_path = (args.output_dir / "overview_mean_f1.png")
    plt.savefig(mean_plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"💾 Saved figure to {mean_plot_path}", flush=True)
    
    # Plot F1 for each model and image
    plt.figure(figsize=(8, 5))
    for model_name, per_image_df in per_image_results.items():
        plt.plot(
            per_image_df["ImageIndex"],
            per_image_df["F1"],
            label=model_name,
        )

    plt.xlabel("Image index")
    plt.ylabel("F1 score")
    plt.title("F1 score per test image")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    per_image_plot_path = (args.output_dir / "f1_per_test_image.png")
    plt.savefig(per_image_plot_path, dpi=200, bbox_inches="tight")
    plt.close()    
    print(f"💾 Saved figure to {per_image_plot_path}", flush=True)
    
    print(f"✅ Done.", flush=True)


if __name__ == "__main__":
    main()
