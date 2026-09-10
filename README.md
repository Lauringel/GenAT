# GenAT

Code for the paper **"Allure of Craquelure: A Variational-Generative Approach to Crack Detection in Paintings"**.

📄 Paper: [arXiv:2602.09730](https://arxiv.org/abs/2602.09730)

## Overview

This repository provides the implementation used in the paper, which formulates crack detection in digitized paintings as an inverse problem. A deep generative prior for the underlying artwork is combined with a Mumford-Shah-inspired variational model for the crack structure, jointly optimized to produce a pixel-level crack map.

<p align="center">
  <img width="400" alt="Input painting detail" src="https://github.com/user-attachments/assets/1a91a21d-4c3f-454e-b705-ad6daa8e21b1" />
  <img width="400" alt="Detected cracks" src="https://github.com/user-attachments/assets/fed01b1b-57c9-4933-aac8-eb54e5396de3" />
</p>
<p align="center">
  <em>Cropped detail of <strong>Hagar und Ismael in der Wüste</strong> by Claude Lorrain (Claude Gellée), 1668, courtesy Bayerische Staatsgemäldesammlungen – Alte Pinakothek München (left), URL: https://www.sammlung.pinakothek.de/de/artwork/ApL8qq2GN2, and the detected crack map (right).</em>
</p>

## Installation

Requires Python 3.11+ (older Python builds can hit an OpenSSL version mismatch when installing dependencies).

```bash
git clone https://github.com/Lauringel/GenAT.git
cd GenAT
pip install -r requirements.txt
```

## Model Checkpoints

This project builds on two third-party model architectures, each handled differently:

**VQGAN** ([taming-transformers](https://github.com/CompVis/taming-transformers), MIT licensed). `pip install git+https://...` for this repo is unreliable — its old `setup.py` can silently install an empty package under some pip/setuptools versions. Clone it as a sibling of this repo instead:
```bash
git clone https://github.com/CompVis/taming-transformers.git ../taming-transformers
```
That sibling location (`../taming-transformers`) is assumed by default — no extra configuration needed if you clone it there. If your checkout lives elsewhere, set `TAMING_TRANSFORMERS_PATH` to point at it.

Our fine-tuned VQGAN checkpoints are redistributed on Zenodo under CC BY 4.0 and are downloaded and cached automatically the first time a script needs them (see `checkpoints.py`). <!-- Pass `--vqgan-checkpoint` to use a local checkpoint instead. -->
The L1 checkpoint is used for GenAT with \(R_{\mathrm{PReg}}\), while the L2 checkpoint is used for GenAT without \(R_{\mathrm{PReg}}\). During grid search, a local fine-tuned checkpoint can instead be supplied using `--vqgan-finetuned`.
The original ImageNet VQGAN checkpoint is only required for running with `USE_FINETUNED=False` or for fine-tuning VQGAN again. In these cases, supply it using `--vqgan-pretrained` or `--pretrained-checkpoint`, respectively.

**DeepCrack** ([original repo](https://github.com/qinnzou/DeepCrack)). We fine-tuned DeepCrack ourselves; although the original training data is restricted to non-commercial research use, the DeepCrack authors have explicitly granted permission to redistribute our fine-tuned checkpoint under CC BY 4.0. It is redistributed on Zenodo and downloaded/cached automatically the same way as the VQGAN checkpoint. <!-- Pass `--deepcrack-ckpt` to use a local checkpoint instead. -->
During grid search, a local fine-tuned checkpoint can instead be supplied using `--deepcrack-checkpoint`. For evaluation, use `--deepcrack-finetuned-checkpoint`. The original `DeepCrack_CT260_FT1.pth` checkpoint is not redistributed here and must be obtained from the original DeepCrack repository.

The DeepCrack code itself (not the weights) is not redistributed — clone the original repo as a sibling of this one:
```bash
git clone https://github.com/qinnzou/DeepCrack.git ../DeepCrack
```
The `codes/model` subfolder is where the importable `deepcrack` module lives, and that sibling location (`../DeepCrack/codes/model`) is assumed by default — no extra configuration needed if you clone it there. If your checkout lives elsewhere, override it with:
```bash
export DEEPCRACK_CODE_PATH=/path/to/your/DeepCrack/codes/model
```

See `model_loading.py` and `checkpoints.py` for the implementation.

## Repository Structure

| Path | Description |
|---|---|
| `crackdetector.py`, `loss_functions.py`, `optim.py` | Core GenAT model, objective, and latent/phase-field optimization. |
| `ambrosio_tortorelli.py` | Classical and masked Ambrosio–Tortorelli baselines. |
| `model_loading.py`, `checkpoints.py` | Loading third-party architectures and resolving checkpoints from Zenodo or local paths. |
| `data_loading.py`, `preprocessing.py`, `postprocessing.py` | Data loading, CLAHE preprocessing, thresholding, tiling, and stitching. |
| `gridsearch.py`, `evaluation.py` | Validation-based hyperparameter selection and test-set evaluation. |
| `finetune_VQGAN.py`, `finetune_DeepCrack.py` | Fine-tuning scripts for the generative model and crack prior. |
| `synthetic_data.py`, `plots_paper.py` | Synthetic-image generation and paper-figure creation. |
| `configs/`, `pictures/`, `notebooks/` | VQGAN configuration, included example images, and interactive examples. |

## Scripts

### `synthetic_data.py`

Creates a single synthetic cracked painting by overlaying a binary crack mask onto a crack-free painting image. Both inputs are resized to 512x512, and the blending direction (dark cracks over a bright painting, or bright cracks over a dark one) is chosen automatically from the painting's mean brightness.

```bash
python synthetic_data.py --painting path/to/painting.jpg --mask path/to/mask.png --out-dir path/to/output
```

| Argument | Description |
|---|---|
| `--painting` | Path to the crack-free painting image. |
| `--mask` | Path to the binary crack mask (0/255) to overlay. |
| `--out-dir` | Directory the outputs are written to. |

Writes `synthetic_single.png` (the synthetic cracked painting) and `mask_single.png` (the resized binary mask) to `--out-dir`.

### Hyperparameter selection (`gridsearch.py`)

Run `gridsearch.py` three times using the switches near the beginning of the
script:

| Experiment | Settings |
|---|---|
| Masked AT | `USE_VQGAN=False`, `SimpleMaskedAT=True` |
| GenAT with \(R_{\mathrm{PReg}}\) | `USE_VQGAN=True`, `USE_PReg=True` |
| GenAT without \(R_{\mathrm{PReg}}\) | `USE_VQGAN=True`, `USE_PReg=False` |

Stage 1 uses the first five validation images in sorted order. For GenAT,
stage 2 evaluates the DeepCrack-prior weights on the full validation set.

For GenAT:

```bash
python gridsearch.py \
  --val-images path/to/validation/images \
  --val-masks path/to/validation/masks \
  --vqgan-pretrained path/to/original_vqgan.ckpt
```

For Masked AT, omit `--vqgan-pretrained`.

Precomputed grid-search results are provided as CSV files in `results/gridsearch`. They contain the tested hyperparameter values and corresponding validation metrics for GenAT with and without \(R_{\mathrm{PReg}}\), as well as for the masked Ambrosio–Tortorelli baseline (Masked AT).
Each row represents one evaluated combination of \(\mu\), \(\varepsilon\), \(\lambda_{\mathrm{AT}}\), and thresholding method and reports the corresponding metrics, macro-averaged over the first five validation images.

### Test-set evaluation (`evaluation.py`)

After generating all three sets of validation CSV files:

```bash
python evaluation.py \
  --test-images path/to/test/images \
  --test-masks path/to/test/masks \
  --deepcrack-checkpoint checkpoints/DeepCrack_CT260_FT1.pth
```

Results are written to `results/evaluation`.

### Reproduce the paper figures

The paper figures can be reproduced using `plots_paper.py` or the interactive
notebook `notebooks/02_reproduce_paper_plots.ipynb`.

In either version, select the GenAT variant with or without
\(R_{\mathrm{PReg}}\), then run the script or all notebook cells. The resulting
comparison figure, soft crack maps, Otsu-thresholded binary maps, and overlays
are saved to `results/plots_paper`.

## Citation

If you use this code, please cite the paper:

```bibtex
@article{paul2026allure,
  title   = {Allure of Craquelure: A Variational-Generative Approach to Crack Detection in Paintings},
  author  = {Paul, Laura and Rauhut, Holger and Burger, Martin and Kabri, Samira and Roith, Tim},
  journal = {arXiv preprint arXiv:2602.09730},
  year    = {2026}
}
```

## License

Except where otherwise noted, this repository and the redistributed fine-tuned checkpoints are licensed under CC BY 4.0. `Brugghen_patch.jpg` and `Lorrain_patch.jpg` are licensed under CC BY-SA 4.0 (see `pictures/README.md` for attribution).
