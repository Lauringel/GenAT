"""
Construction and weight-loading for the two third-party model architectures used in this
project: VQGAN (taming-transformers) and DeepCrack.

This module only builds `nn.Module`s and loads a given checkpoint path into them — it does
not know where checkpoint files come from. See checkpoints.py for resolving/downloading the
actual checkpoint paths.
"""

import collections.abc
import os
import sys
import types

import torch
from omegaconf import OmegaConf

from checkpoints import DEEPCRACK_REPO_URL

TAMING_TRANSFORMERS_PATH_ENV = "TAMING_TRANSFORMERS_PATH"
DEEPCRACK_CODE_PATH_ENV = "DEEPCRACK_CODE_PATH"

# Default assumption for both: a sibling checkout next to this repo, e.g.
#   git clone https://github.com/CompVis/taming-transformers.git ../taming-transformers
#   git clone https://github.com/qinnzou/DeepCrack.git ../DeepCrack
# `pip install git+...` for taming-transformers is unreliable (it can silently install an
# empty package due to how its old setup.py interacts with PEP 517 build isolation), so a
# local checkout added to sys.path is the recommended path for both dependencies.
# Override with TAMING_TRANSFORMERS_PATH / DEEPCRACK_CODE_PATH if yours live elsewhere.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TAMING_TRANSFORMERS_PATH = os.path.normpath(
    os.path.join(_REPO_ROOT, "..", "taming-transformers")
)
DEFAULT_DEEPCRACK_CODE_PATH = os.path.normpath(
    os.path.join(_REPO_ROOT, "..", "DeepCrack", "codes", "model")
)


def _ensure_torch_six_shim():
    """pytorch-lightning==1.0.8 imports names from torch._six, removed in PyTorch 1.13+.
    We keep a modern torch (needed for Python 3.11 support, which conflicts with the old
    torch versions that still have torch._six) and shim the handful of names old code
    expects instead. Extend this if a later import needs another name from it."""
    if "torch._six" in sys.modules:
        return
    six_shim = types.ModuleType("torch._six")
    six_shim.string_classes = (str,)
    six_shim.int_classes = (int,)
    six_shim.container_abcs = collections.abc
    six_shim.inf = float("inf")
    sys.modules["torch._six"] = six_shim


def _import_vqmodel():
    _ensure_torch_six_shim()

    taming_path = os.environ.get(TAMING_TRANSFORMERS_PATH_ENV, DEFAULT_TAMING_TRANSFORMERS_PATH)
    if taming_path and taming_path not in sys.path:
        sys.path.append(taming_path)

    try:
        from taming.models.vqgan import VQModel
    except ImportError as e:
        raise ImportError(
            "Could not import taming-transformers. Clone it into a sibling folder "
            f"(expected at {DEFAULT_TAMING_TRANSFORMERS_PATH}): "
            "`git clone https://github.com/CompVis/taming-transformers.git ../taming-transformers`, "
            f"or set the {TAMING_TRANSFORMERS_PATH_ENV} environment variable to point at your own checkout."
        ) from e
    return VQModel


def _import_deepcrack():
    code_path = os.environ.get(DEEPCRACK_CODE_PATH_ENV, DEFAULT_DEEPCRACK_CODE_PATH)
    if code_path and code_path not in sys.path:
        sys.path.append(code_path)

    try:
        from deepcrack import DeepCrack
    except ImportError as e:
        raise ImportError(
            f"Could not import DeepCrack. Clone {DEEPCRACK_REPO_URL} into a sibling folder "
            f"(expected at {DEFAULT_DEEPCRACK_CODE_PATH}), or set the "
            f"{DEEPCRACK_CODE_PATH_ENV} environment variable to point at your own checkout."
        ) from e
    return DeepCrack


def load_vqgan_model_ckpt(config_path, checkpoint_path, device):
    """Build a VQModel from `config_path` and load weights from `checkpoint_path`."""
    VQModel = _import_vqmodel()

    config = OmegaConf.load(config_path)
    model = VQModel(**config.model.params).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        print("Loaded Lightning checkpoint (using state_dict).", flush=True)
    else:
        state_dict = checkpoint
        print("Loaded raw state_dict (no Lightning wrapper).", flush=True)

    clean_state = {k.replace("model.", ""): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    print("Missing keys:", missing, flush=True)
    print("Unexpected keys:", unexpected, flush=True)

    model.eval()
    print("VQGAN model loaded successfully.", flush=True)
    return model


def load_dcmodel(path, device, is_checkpoint=False):
    """Build a DeepCrack model and load weights from `path`, frozen and in eval mode."""
    DeepCrack = _import_deepcrack()

    model = DeepCrack().to(device)
    state = torch.load(path, map_location=device)

    if is_checkpoint and "model" in state:
        state = state["model"]
        print(f"Loaded fine-tuned checkpoint from {path}", flush=True)
    elif "state_dict" in state:
        state = state["state_dict"]
        print(f"Loaded state_dict from {path}", flush=True)
    else:
        print(f"Loaded raw model weights from {path}", flush=True)

    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    print("DeepCrack loaded and frozen.", flush=True)
    return model
