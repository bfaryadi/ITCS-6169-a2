# Self-supervised representation learning on 2,400 images

ITCS 6169/8169 Computer Vision, Assignment 2. Everything is trained **from scratch on the 2,400 provided training images**: no pretrained weights and no external data. Four methods are compared at 64px and 128px input resolution, with the Assignment 1 supervised CNN as the reference:

| Method | Backbone | How the representation is learned |
|---|---|---|
| SigLIP-style CNN | A1 CNN | two-view contrastive pretraining with a pairwise sigmoid loss, then supervised fine-tuning |
| BYOL CNN | A1 CNN | two-view self-distillation (online/target networks, no negatives), then supervised fine-tuning |
| Supervised ViT | small ViT | supervised training only |
| ViT + SSAT | small ViT | joint training: classification + masked-patch reconstruction on a shared encoder |

## Best model and how to reproduce the reported number

Best checkpoint: `checkpoints/128px_siglip_cnn_final_s0_finetune.pt`

```bash
python evaluate.py --checkpoint checkpoints/128px_siglip_cnn_final_s0_finetune.pt --data-dir data/test
# expected output includes:  Test accuracy: 0.8750 (350/400)
```

`evaluate.py` rebuilds the network from `models.py`, loads the weights of the best-validation epoch (`--weights best`, the default and the setting used for all reported numbers), and evaluates on the test folder. Each checkpoint file also contains the last-epoch weights (`--weights last`).

## Repository layout

```
models.py                 model definitions (CNN, SSL heads and losses, BYOL pieces, ViT/SSAT)
evaluate.py               evaluation and single-image inference
notebooks/                training code (run in the order below)
  ITCS_6169_a2_siglip_cnn.ipynb         SigLIP-style CNN, 64px: ablations, final runs, fine-tuning
  ITCS_6169_a2_byol_cnn.ipynb           BYOL CNN, 64px
  ITCS_6169_a2_vit_ssat.ipynb           supervised ViT and ViT + SSAT, 64px (v1 recipe)
  ITCS_6169_a2_vit_ssat_v2.ipynb        second recipe-search round for the ViTs
  ITCS_6169_a2_*_128px.ipynb            128px finals (reuse the 64px configs, no ablations)
  ITCS_6169_a2_results_summary.ipynb    re-evaluates every checkpoint and builds the results tables
checkpoints/              trained weights and the chosen configurations (see below)
```

## Setup

- Python 3.11 was used for the reported runs. Packages: `torch`, `torchvision`, `pillow`, `numpy`; the notebooks also use `matplotlib`.
- The training runs used an AMD GPU on Windows through `torch-directml`. `evaluate.py` picks CUDA, then DirectML, then CPU automatically (override with `--device cpu|cuda|dml`). CPU is enough for evaluation.
- Data (provided by the course, not redistributed here), in torchvision `ImageFolder` layout:

```
data/train/<class_name>/*.jpg     2,400 images, 16 classes
data/test/<class_name>/*.jpg      test images
```

## Evaluating any checkpoint

Checkpoint files are named `<prefix><label>_final_s<seed>_<stage>.pt`. The prefix is empty for 64px and `128px_` for 128px runs. Use the `_finetune.pt` file for the CNNs and the `_train.pt` file for the ViTs.

| Row in the results table | Files (seeds 0 and 1) |
|---|---|
| SigLIP-style CNN | `siglip_cnn_final_s{0,1}_finetune.pt`, `128px_siglip_cnn_final_s{0,1}_finetune.pt` |
| BYOL CNN | `byol_cnn_final_s{0,1}_finetune.pt`, `128px_byol_cnn_final_s{0,1}_finetune.pt` |
| Supervised ViT | `vit_sup_final_s{0,1}_train.pt`, `128px_vit_sup_final_s{0,1}_train.pt` |
| ViT + SSAT | `vit_ssat_final_s{0,1}_train.pt`, `128px_vit_ssat_final_s{0,1}_train.pt` |

The `*_pretrain.pt` files (SSL encoder plus projection head) are only needed to repeat the fine-tuning step and are not classifiers.

```bash
# image size: ViT checkpoints carry it; for CNNs it is read from the 128px_ prefix, or pass --img-size
python evaluate.py --checkpoint checkpoints/siglip_cnn_final_s0_finetune.pt --data-dir data/test
python evaluate.py --checkpoint checkpoints/128px_vit_ssat_final_s0_train.pt --per-class

# validation split, re-created with the training seed (fixed split: 1,920 train / 480 validation)
python evaluate.py --checkpoint checkpoints/siglip_cnn_final_s0_finetune.pt --split val --train-dir data/train

# classify one image
python evaluate.py --checkpoint checkpoints/siglip_cnn_final_s0_finetune.pt --image path/to/image.jpg
```

For `--split val` with `--weights best`, the script prints the validation accuracy logged during training next to the recomputed one; they should agree, which confirms that the split and preprocessing match training. `python models.py` prints the parameter count of every model.

`checkpoints/` also holds the chosen configurations (`siglip_final_cfg.json`, `byol_final_cfg.json`, `vit_final_cfgs.json`), the ablation tables (`*_ablation_*ep.json`) and, after running the summary notebook, `results_per_model.csv` and `results_summary.json`.

## Training

Run the notebooks from the repository root so that `data/` and `checkpoints/` resolve. Finished runs are written to `checkpoints/` and skipped on re-run (a crash can be resumed). Set `SMOKE_TEST = True` in the first config cell to check a notebook end to end in a few minutes.

1. **SigLIP-style CNN (64px).** One-at-a-time ablations (50 SSL epochs, scored by a linear probe on frozen features), a combined-config check, then final runs: 200 SSL epochs, fine-tuning, 2 seeds. Writes `siglip_final_cfg.json`.
2. **BYOL CNN (64px).** Starts from the shared augmentation and optimizer settings in `siglip_final_cfg.json`, so run it second. Same protocol. Writes `byol_final_cfg.json`.
3. **ViTs (64px).** Learning-rate search on the supervised ViT, SSAT knob ablations (60 epochs), then 100-epoch finals for both models, 2 seeds. Writes `vit_final_cfgs.json`. The v2 notebook runs the second recipe-search round.
4. **128px notebooks.** Finals only: they load the 64px configurations and retrain at 128px. The SSL batch size is reduced to 64 and SSL runs for 100 epochs to fit in GPU memory. They require the 64px notebooks to have been run in the same folder.
5. **Results summary.** Re-evaluates every checkpoint (best and last weights) at both resolutions and writes the tables and chart.

Measured speed on the training machine (64px): about 3 s per BYOL SSL epoch (about 10 min for 200 epochs), about 6 s per supervised ViT epoch and 11 s per SSAT epoch. Times differ on other hardware, and 128px runs are several times slower.

## Final hyperparameters

**Fine-tuning (all CNN rows, same as Assignment 1).** AdamW, lr 5e-4, weight decay 0.05, cosine schedule, batch 32, 100 epochs, mixup 0.4 and CutMix 1.0 (CutMix with probability 0.5, at most one per batch), random flip, light color jitter and random grayscale. The deployed network is the A1 CNN: conv stack, global average pool, dropout 0.3, Linear(256, 16). The SSL projection head and predictor are discarded after pretraining.

**SigLIP-style SSL.** Loss `-(1/n) sum log sigmoid(label * (t * z1_i . z2_j + b))` over all cross-view pairs in the batch (label +1 for the two views of the same image, -1 otherwise), with `t = exp(log_t)` and `b` learnable. Initial values: t = 10 and b = -15. Batch 128, lr 5e-4, weight decay 0.05, AdamW with cosine decay, projection head 256 -> 512 -> BatchNorm -> ReLU -> 128, view augmentation: random-resized crop (min scale 0.3), flip, color jitter 0.4 (applied with p = 0.8), grayscale p = 0.2. 200 epochs at 64px, 100 epochs and batch 64 at 128px.

**BYOL.** Online network = A1 encoder + projector (256 -> 512 -> 128) + predictor (128 -> 512 -> 128); target = EMA copy of encoder + projector, no predictor, no gradients. Loss `2 - 2 cos(prediction, target)` symmetrised over the two views. EMA momentum starts at 0.9 and rises to 1 on a cosine schedule (the target is updated every 4 steps with the momentum raised to the 4th power). Batch 64, lr 2.5e-4, weight decay 0.05, same augmentations as above. 200 epochs at 64px, 100 at 128px.

**ViT and SSAT.** Encoder: patch embedding, CLS token, learned position embeddings, 6 pre-norm blocks (width 192, 3 heads, MLP ratio 4, dropout 0.1), LayerNorm, Linear(192, 16) on the CLS token; about 2.7M parameters at 64px (patch 8) and 2.8M at 128px (patch 16, so both use 64 tokens). SSAT adds a training-only decoder (width 128, depth 2, 4 heads) that reconstructs the hidden patches of a second augmented view in which only 25% of patches are visible (MSE on per-patch-normalised pixels); loss = 0.5 * cross-entropy + 0.5 * reconstruction. The supervised ViT is the same model with lambda = 1. AdamW, lr 2.5e-4, weight decay 0.05 (weights only), batch 64, 5 warm-up epochs then cosine decay, 100 epochs; mixup and CutMix as above on the classification view.

## Model selection protocol

The 2,400 training images are split once with a fixed seed (1,920 train / 480 validation). Hyperparameters were chosen on validation only: one knob at a time from a base configuration, and a change was adopted only if it beat the base by at least 2 points (validation noise is about 1.5 points; the measured seed-to-seed gap was larger for some runs). The combination of adopted knobs was re-run and kept only if it was at least as good as the base. The test set was evaluated once, after all choices were made. Reported validation accuracy is the best epoch and is therefore optimistic; the test number is the honest estimate. Differences of 2 to 3 points between methods are within noise (2 seeds per method).

## Notes

- No pretrained weights or external data are used. The ImageNet mean and standard deviation appear in the code only as fixed input-normalisation constants.
- Training is seeded, but GPU kernels (especially on DirectML) are not fully deterministic, so re-running a notebook will not reproduce the stored weights bit for bit. Evaluating the stored checkpoints is deterministic.
- The notebooks contain workarounds for DirectML (for example building some tensors on the CPU); they are harmless on CUDA or CPU.

## AI assistance

Code was developed with AI coding assistance (Claude), as permitted by the assignment. The choice of methods, the experiments, and the interpretation of results are the author's.
