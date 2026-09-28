"""
Model checkpoint resolution.

VQGAN: fine-tuned by us from the MIT-licensed CompVis/taming-transformers VQGAN.
DeepCrack: fine-tuned by us from the third-party baseline (https://github.com/qinnzou/DeepCrack),
redistributed with the original authors' explicit permission.

Both fine-tuned checkpoints are redistributed on Zenodo under CC BY 4.0 and are downloaded
and cached automatically the first time a script needs them.
"""

import os
import urllib.request

# Published Zenodo records hosting the model checkpoints.
ZENODO_BASE_URL = "https://zenodo.org"
VQGAN_ZENODO_RECORD_ID = "21258301"
DEEPCRACK_ZENODO_RECORD_ID = "22968130"

DEEPCRACK_REPO_URL = "https://github.com/qinnzou/DeepCrack"

# Both VQGAN checkpoints are used for different experiments in the paper.
VQGAN_FINETUNED_L1 = "vqgan_finetuned_l1_large_2.pth"
VQGAN_FINETUNED_L2 = "vqgan_finetuned_l2_large.pth"
DEEPCRACK_FINETUNED = "deepcrack_finetuned_synthetic_unfreeze.pth"
DEEPCRACK_PRETRAINED = "DeepCrack_CT260_FT1.pth"


def _report_progress(block_num, block_size, total_size):
    if total_size <= 0:
        return
    downloaded = block_num * block_size
    pct = min(100, downloaded * 100 // total_size)
    print(f"\r  {pct}%", end="", flush=True)


def _get_checkpoint(record_id, filename, cache_dir):
    """Return the local path to a checkpoint, downloading it from Zenodo if needed."""
    os.makedirs(cache_dir, exist_ok=True)
    out_path = os.path.join(cache_dir, filename)

    if not os.path.exists(out_path):
        url = f"{ZENODO_BASE_URL}/records/{record_id}/files/{filename}?download=1"
        tmp_path = out_path + ".part"
        print(f"Downloading {filename} from Zenodo...")
        # Download to a temp path first so an interrupted download can't be mistaken
        # for a complete, cached checkpoint on the next run.
        urllib.request.urlretrieve(url, tmp_path, reporthook=_report_progress)
        os.replace(tmp_path, out_path)
        print()

    return out_path


def get_vqgan_checkpoint(filename, cache_dir="checkpoints"):
    return _get_checkpoint(VQGAN_ZENODO_RECORD_ID, filename, cache_dir)


def get_deepcrack_checkpoint(filename=DEEPCRACK_FINETUNED, cache_dir="checkpoints"):
    return _get_checkpoint(DEEPCRACK_ZENODO_RECORD_ID, filename, cache_dir)
