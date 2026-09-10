from pathlib import Path

import numpy as np
import torch
import random
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_SIZE = (512, 512)

image_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])


SUPPORTED_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
}

def find_images(directory):
    directory = Path(directory)

    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def load_rgb_image(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    return Image.open(path).convert("RGB")


def load_binary_mask(path, image_size=IMAGE_SIZE):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Mask not found: {path}")

    mask = Image.open(path).convert("L")

    if image_size is not None:
        mask = mask.resize(
            (image_size[1], image_size[0]),
            resample=Image.NEAREST,
        )

    return (np.asarray(mask) > 128).astype(np.float32)


class CrackDataset(Dataset):
    """
    Paired synthetic images and binary crack masks.

    Images and masks are paired according to their sorted filenames.
    """

    def __init__(self, image_dir, mask_dir, augment=False, normalize=False):
        self.images = find_images(image_dir)
        self.masks = find_images(mask_dir)
        self.augment = augment
        self.normalize = normalize

        if len(self.images) != len(self.masks):
            raise ValueError(
                "Image/mask count mismatch: "f"{len(self.images)} images and "f"{len(self.masks)} masks.")

        if not self.images:
            raise ValueError(f"No PNG or JPG images found in {image_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        mask_path = self.masks[index]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = image.resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), resample=Image.BILINEAR)
        mask = mask.resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), resample=Image.NEAREST)
        
        if self.augment:
            if random.random() > 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        
            if random.random() > 0.5:
                image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        
            if random.random() > 0.7:
                angle = random.choice([90, 180, 270])
                image = image.rotate(angle)
                mask = mask.rotate(angle)

        image_tensor = transforms.functional.to_tensor(image)

        if self.normalize:
            image_tensor = transforms.functional.normalize(
                image_tensor,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )

        mask_array = np.asarray(mask)

        # Preserve the existing ground-truth thresholding.
        mask_binary = (mask_array > 128).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0)

        return image_tensor, mask_tensor, image_path.name


def get_loader(image_dir, mask_dir, batch_size=1):
    dataset = CrackDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        augment=False,
        normalize=False
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )


def get_val_loader(image_dir, mask_dir, batch_size=1):
    return get_loader(
        image_dir=image_dir,
        mask_dir=mask_dir,
        batch_size=batch_size,
    )


def get_test_loader(image_dir, mask_dir, batch_size=1):
    return get_loader(
        image_dir=image_dir,
        mask_dir=mask_dir,
        batch_size=batch_size,
    )
