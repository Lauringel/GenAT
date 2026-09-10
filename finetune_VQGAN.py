"""
Fine-tuning of the pretrained VQGAN model on crack-free painting patches.
l2: variant without the mixed-gradient term
l1: variant used with the mixed-gradient term
"""

import os
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import vgg16

from checkpoints import VQGAN_FINETUNED_L1, VQGAN_FINETUNED_L2
from model_loading import load_vqgan_model_ckpt

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ============================================================
# Switches
# ============================================================

BATCH_SIZE = 8
EPOCHS_FINE_G = 25
LR_G = 5e-5
weight_ft_rec = 0.65
weight_ft_perc = 0.25

# ============================================================
# Paths and device
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REPO_ROOT = Path(__file__).resolve().parent
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune VQGAN on crack-free painting patches.")
    parser.add_argument("--nocrack-images", type=Path, required=True)
    parser.add_argument("--pretrained-checkpoint", type=Path, required=True)
    parser.add_argument("--vqgan-config", type=Path, default=REPO_ROOT / "configs" / "vqgan_config.yaml")
    parser.add_argument("--checkpoint-dir", type=Path, default=REPO_ROOT / "checkpoints")
    parser.add_argument("--output-checkpoint", type=Path, default=None)
    parser.add_argument("--reconstruction-loss", choices=["l1", "l2"], default="l2")
    return parser.parse_args()
    

# ============================================================
# Dataset for training
# ============================================================

transform_finetune = transforms.Compose([
    transforms.Resize(520),
    transforms.RandomCrop(512),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.03, contrast=0.03),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5) # normalize to [-1,1] for VQGAN
])

def verify_image(path):
    try:
        Image.open(path).verify()
        return True
    except Exception as e:
        print(f"⚠ Corrupted: {path} ({e})", flush = True)
        return False

class SimpleImageFolder(Dataset):
    def __init__(self, folder, transform=None):
        self.paths = sorted([os.path.join(folder, f) for f in os.listdir(folder)
                             if f.lower().endswith((".jpg",".jpeg",".png",".tif",".tiff"))])
        self.paths = [p for p in self.paths if verify_image(p)]
        self.transform = transform
        print(f"Loaded {len(self.paths)} from {folder}", flush = True)
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform: img = self.transform(img)
        return img


# ============================================================
# Utilities
# ============================================================
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

class VGGPerceptual(nn.Module):
    """Perceptual loss used only for VQGAN fine-tuning."""
    def __init__(self, device):
        super().__init__()
        vgg = vgg16(weights="IMAGENET1K_V1").features[:16].to(device).eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
    def forward(self, x, y):
        # x,y in [-1,1] -> [0,1]
        x = (x + 1) / 2
        y = (y + 1) / 2
        # normalize to ImageNet
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        y = (y - IMAGENET_MEAN) / IMAGENET_STD
        return F.l1_loss(self.vgg(x), self.vgg(y))

# ============================================================
# Fine-tuning specifics
# ============================================================
def finetune_vqgan(V, train_loader, val_loader, epochs, lr, device, reconstruction_loss, output_checkpoint):
    print("Starting VQGAN finetuning on no-crack images...", flush=True)
    best_state = None

    # early stopping parameters
    best_val = float("inf")
    patience = 3 # stop finetuning if validation loss doesn’t improve for patience epochs
    patience_ctr = 0

    # losses
    if reconstruction_loss == "l1":
        reconstruction_criterion = nn.L1Loss()
    elif reconstruction_loss == "l2":
        reconstruction_criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unknown reconstruction loss: {reconstruction_loss}")
    percept = VGGPerceptual(device)

    
    # ---------------------------
    # Freeze encoder & codebook
    # ---------------------------
    for p in V.encoder.parameters(): 
        p.requires_grad = False
    for block in V.encoder.down[-2:]:
        for p in block.parameters():
            p.requires_grad = True
    for p in V.quantize.parameters(): 
        p.requires_grad = False
    for p in V.quant_conv.parameters(): 
        p.requires_grad = False

    
    # ---------------------------
    # Train decoder + post_quant_conv
    # ---------------------------
    for p in V.decoder.parameters(): 
        p.requires_grad = True
    for p in V.post_quant_conv.parameters(): 
        p.requires_grad = True

    opt = torch.optim.Adam(
        [p for p in V.parameters() if p.requires_grad], lr=lr
    )

    for epoch in range(epochs):
        running = 0.0

        # ---- training ----
        V.train()
        for img in train_loader:
            img = img.to(device)
            opt.zero_grad()

            z_q, _, _ = V.encode(img) # use official encode()/decode() pipeline from vqgan
            rec = V.decode(z_q)
            
            L_rec = reconstruction_criterion(rec, img)
            L_perc = percept(rec, img)
            
            loss = weight_ft_rec * L_rec + weight_ft_perc * L_perc
            loss.backward()
            opt.step()

            running += loss.item()

        print(f"[VQGAN FT] Epoch {epoch + 1}/{epochs} — Loss={running / len(train_loader):.6f}")

        
        # ---- validation ----
        V.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for img in val_loader:
                img = img.to(device)
                
                z_q, _, _ = V.encode(img)
                rec = V.decode(z_q)
                
                L_rec = reconstruction_criterion(rec, img)
                L_perc = percept(rec, img)

                loss = weight_ft_rec * L_rec + weight_ft_perc * L_perc
                val_loss += loss.item()

        val_loss /= len(val_loader)

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            patience_ctr = 0
            best_state = {k: v.detach().cpu().clone() for k, v in V.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print("🛑 Early stopping triggered.")
                break

    
    assert best_state is not None, "No model state saved during finetuning!"
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, output_checkpoint)
    
    print(
        f"Finetuning complete. Saved as {output_checkpoint}",
        flush=True,
    )
    print("Finetuning complete. 💾 Saved finetuned VQGAN as {output_checkpoint}", flush=True)

# ============================================================
# Main: Fine-tuning of the VQGAN model
# ============================================================
if __name__ == "__main__":
    args = parse_args()
    
    checkpoint_filename = (VQGAN_FINETUNED_L1 if args.reconstruction_loss == "l1" else VQGAN_FINETUNED_L2)

    output_checkpoint = (args.output_checkpoint if args.output_checkpoint is not None else args.checkpoint_dir / checkpoint_filename)
    
    nocrack_dataset = SimpleImageFolder(args.nocrack_images, transform=transform_finetune)

    train_size = int(0.9 * len(nocrack_dataset))
    val_size = len(nocrack_dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(nocrack_dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False)


    vqgan = load_vqgan_model_ckpt(
        args.vqgan_config,
        args.pretrained_checkpoint,
        DEVICE
    )

    finetune_vqgan(
        vqgan,
        train_loader,
        val_loader,
        EPOCHS_FINE_G,
        LR_G,
        DEVICE,
        reconstruction_loss=args.reconstruction_loss,
        output_checkpoint=output_checkpoint
    )
