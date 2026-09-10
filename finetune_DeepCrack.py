"""
Fine-tuning of the pretrained DeepCrack model on synthetic images.
"""

import argparse
import glob
import os
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

from checkpoints import DEEPCRACK_PRETRAINED, DEEPCRACK_FINETUNED, get_deepcrack_checkpoint
from model_loading import load_dcmodel
from data_loading import CrackDataset

num_epochs = 25
batch_size = 8
num_workers = 4

# ============================================================
# Paths
# ============================================================
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = REPO_ROOT / "checkpoints" / DEEPCRACK_FINETUNED

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune DeepCrack on synthetic painting cracks.")
    parser.add_argument("--train-images", type=Path, required=True)
    parser.add_argument("--train-masks", type=Path, required=True)
    parser.add_argument("--val-images", type=Path, required=True)
    parser.add_argument("--val-masks", type=Path, required=True)
    parser.add_argument(
        "--pretrained-checkpoint",
        type=Path,
        required=True,
        help="Original pretrained DeepCrack checkpoint.",
    )
    parser.add_argument(
        "--output-checkpoint",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    return parser.parse_args()


# ============================================================
# Helpers
# ============================================================

# Class-balanced BCE loss
def class_balanced_bce_loss(pred, target, eps=1e-8):
    pos = target.sum()
    neg = (1 - target).sum()
    beta = neg / (pos + neg + eps)
    pos_weight = beta / (1 - beta + eps)
    loss = nn.functional.binary_cross_entropy_with_logits(
        pred, target, pos_weight=torch.as_tensor(pos_weight, device=pred.device)
    )
    return loss

# Training and validation function
def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    use_amp = scaler is not None

    for imgs, masks, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=use_amp):
            outs = model(imgs)

            if isinstance(outs, torch.Tensor):
                outs = [outs]

            loss = sum(
                class_balanced_bce_loss(
                    out,
                    F.interpolate(
                        masks,
                        size=out.shape[-2:],
                        mode="nearest",
                    ),
                )
                for out in outs
            )

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    tot_loss, f1_scores = 0, []
    
    for imgs, masks, _ in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        outs = model(imgs)
        
        if isinstance(outs, torch.Tensor):
            out = outs
        elif isinstance(outs, (list, tuple)):
            out = outs[-1]
        else:
            out = list(outs.values())[-1]
        
        loss = class_balanced_bce_loss(out, masks)
        prob = torch.sigmoid(out)
        pred = (prob > 0.5).float()
        tp = (pred * masks).sum()
        fp = (pred * (1 - masks)).sum()
        fn = ((1 - pred) * masks).sum()
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
        tot_loss += loss.item()
        f1_scores.append(f1.item())
    
    return tot_loss / len(loader), np.mean(f1_scores)


# ============================================================
# Main: Fine-tuning of the DeepCrack model
# ============================================================
if __name__ == "__main__":
    args = parse_args()
    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    pretrained_checkpoint = (
        args.pretrained_checkpoint
        if args.pretrained_checkpoint is not None
        else Path(
            get_deepcrack_checkpoint(
                DEEPCRACK_PRETRAINED,
                cache_dir=args.checkpoint_dir,
            )
        )
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    train_ds = CrackDataset(args.train_images, args.train_masks, augment=True, normalize=True)
    val_ds = CrackDataset(args.val_images, args.val_masks, augment=False, normalize=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Build and load model
    model = load_dcmodel(
        args.pretrained_checkpoint,
        device,
        is_checkpoint=False,
    )

    # Freeze backbone initially
    for parameter in model.parameters():
        parameter.requires_grad = False

    for module_name in ("fuse1", "fuse2", "fuse3", "fuse4", "fuse5", "final"):
        for parameter in getattr(model, module_name).parameters():
            parameter.requires_grad = True
            
    print(f"Loading pretrained weights: {args.pretrained_checkpoint}")

    freeze_backbone = True
    print("Backbone frozen for first five epochs.")

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)
    scaler = GradScaler("cuda") if device.type == "cuda" else None

    best_val_f1 = 0
    
    for epoch in range(1, num_epochs + 1):
        # Unfreeze after 5 epochs
        if freeze_backbone and epoch == 6:
            print("Unfreezing backbone.")
            for p in model.parameters():
                p.requires_grad = True
                
            new_batch_size = max(1, batch_size // 2)
            if new_batch_size != batch_size:
            	print(f"Reducing batch_size from {batch_size} to {new_batch_size} after unfreezing.")
            	batch_size = new_batch_size
            	# re-create dataloaders with the new batch size
            	train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
            	val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
            
            # Preserve the historical choice to disable AMP after unfreezing.
            scaler = None
            if device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device)
        val_loss, val_f1 = validate(model, val_loader, device)
        print(f"[Epoch {epoch:03d}] TrainLoss={train_loss:.4f}  ValLoss={val_loss:.4f}  ValF1={val_f1:.4f}")

        # save every epoch
        ckpt_path = args.output_checkpoint.parent / f"epoch_{epoch:03d}.pth"
        torch.save({"epoch": epoch, "model": model.state_dict()}, ckpt_path)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                {"epoch": epoch, "model": model.state_dict()},
                args.output_checkpoint,
            )
            print("✅ Saved new best model.")

    print("\n✅ Fine-tuning complete.")
