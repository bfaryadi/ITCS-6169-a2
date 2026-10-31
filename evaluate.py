#!/usr/bin/env python3
"""Evaluation / inference for ITCS 6169/8169 Assignment 2.

Loads one trained checkpoint (CNN or ViT), rebuilds the network from models.py and reports accuracy on the
test set (default), on the validation split used during training, or predicts a single image.

Examples
--------
    # test-set accuracy of a fine-tuned CNN (image size is read from the file name: 128px_ prefix -> 128)
    python evaluate.py --checkpoint checkpoints/128px_siglip_cnn_final_s0_finetune.pt --data-dir data/test

    # same, with an explicit image size and per-class accuracy
    python evaluate.py --checkpoint checkpoints/128px_siglip_cnn_final_s0_finetune.pt --img-size 128 --per-class

    # a ViT checkpoint (image size, patch size and architecture are read from the checkpoint itself)
    python evaluate.py --checkpoint checkpoints/vit_ssat_final_s0_train.pt

    # validation split (re-created from the training images with the same seed as in training)
    python evaluate.py --checkpoint checkpoints/siglip_cnn_final_s0_finetune.pt --split val --train-dir data/train

    # classify one image
    python evaluate.py --checkpoint checkpoints/128px_siglip_cnn_final_s0_finetune.pt --image some.jpg

Expected folder layout (torchvision ImageFolder): <data-dir>/<class_name>/<image files>.
"""
import argparse
import math
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

from models import TNet, ViTSSAT, vit_kwargs

try:
    import torch_directml  # optional: AMD GPUs on Windows
    HAS_DML = torch_directml.is_available()
except ImportError:
    HAS_DML = False

# Generic input normalisation constants only. No pretrained weights are used anywhere in this project.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# --------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------
def pick_device(name):
    if name == 'cpu':
        return torch.device('cpu')
    if name == 'cuda':
        return torch.device('cuda')
    if name == 'dml':
        if not HAS_DML:
            sys.exit('--device dml requested but torch_directml is not available')
        return torch_directml.device()
    # auto: CUDA, then DirectML, then CPU
    if torch.cuda.is_available():
        return torch.device('cuda')
    if HAS_DML:
        return torch_directml.device()
    return torch.device('cpu')


def load_checkpoint(path):
    """Checkpoints are dicts: best_state, last_state, history, hparams (written by the training notebooks)."""
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except Exception:
        # older PyTorch (no weights_only) or non-tensor objects in the metadata; the file is our own
        return torch.load(path, map_location='cpu', weights_only=False)


def build_model(ck, state, ckpt_path, img_size_arg):
    """Rebuild the network from the checkpoint. Returns (model, img_size, family)."""
    keys = list(state.keys())
    if 'patch_embed.weight' in keys:                                   # ---- ViT (supervised or SSAT) ----
        cfg = ck['hparams']['cfg']
        patch = state['patch_embed.weight'].shape[-1]
        n_tokens = state['pos_embed'].shape[1] - 1
        side = math.isqrt(n_tokens)
        img = side * patch
        if img_size_arg is not None and img_size_arg != img:
            sys.exit(f'--img-size {img_size_arg} does not match this ViT checkpoint (trained at {img}px)')
        kwargs = vit_kwargs(cfg)
        kwargs['patch'] = patch
        model = ViTSSAT(img=img, num_classes=state['head.weight'].shape[0], **kwargs)
        model.load_state_dict(state)
        return model, img, 'ViT'
    if 'head.0.weight' in keys and 'classifier.3.weight' not in keys:
        sys.exit('This is an SSL *pretraining* checkpoint (encoder + projection head), not a classifier. '
                 'Use the *_finetune.pt file of the same run.')
    if 'classifier.3.weight' in keys:                                   # ---- CNN (A1 architecture) ----
        img = img_size_arg
        if img is None:
            m = re.match(r'(\d+)px_', Path(ckpt_path).name)
            img = int(m.group(1)) if m else 64
            print(f'[info] --img-size not given -> using {img} '
                  f'({"from the file-name prefix" if m else "the default for files without a prefix"})')
        model = TNet(num_classes=state['classifier.3.weight'].shape[0])
        model.load_state_dict(state)
        return model, img, 'CNN'
    sys.exit('Unrecognised checkpoint contents (expected a TNet or ViTSSAT state dict)')


def make_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class ValSplit(Dataset):
    """Validation images, re-created exactly as in training: seeded random_split of the training folder,
    each image first resized with PIL bilinear, then the evaluation transform."""
    def __init__(self, base, indices, img_size, tf):
        self.base, self.indices, self.img_size, self.tf = base, indices, img_size, tf

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, y = self.base[self.indices[i]]
        return self.tf(img.resize((self.img_size, self.img_size), Image.BILINEAR)), y


def class_names_from_dir(d):
    d = Path(d)
    return sorted(p.name for p in d.iterdir() if p.is_dir()) if d.is_dir() else None


@torch.no_grad()
def run_eval(model, loader, device, num_classes):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    pc_correct, pc_total = [0] * num_classes, [0] * num_classes
    for x, y in loader:
        logits = model(x.to(device)).float().cpu()
        loss_sum += F.cross_entropy(logits, y, reduction='sum').item()
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += len(y)
        for t, p in zip(y.tolist(), pred.tolist()):
            pc_total[t] += 1
            pc_correct[t] += int(t == p)
    return correct / total, loss_sum / total, correct, total, pc_correct, pc_total


# --------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', required=True, help='path to a *_finetune.pt (CNN) or *_train.pt (ViT) file')
    ap.add_argument('--weights', choices=['best', 'last'], default='best',
                    help="'best' = weights of the best-validation epoch (used for the reported results); "
                         "'last' = weights after the final epoch")
    ap.add_argument('--split', choices=['test', 'val'], default='test')
    ap.add_argument('--data-dir', default='data/test', help='ImageFolder with the test images')
    ap.add_argument('--train-dir', default='data/train', help='ImageFolder with the training images '
                                                              '(needed for --split val and for class names)')
    ap.add_argument('--image', help='classify a single image instead of evaluating a split')
    ap.add_argument('--img-size', type=int, default=None,
                    help='input resolution; CNN only (inferred from the file name or defaults to 64). '
                         'ViT checkpoints carry their own resolution.')
    ap.add_argument('--batch-size', type=int, default=100)
    ap.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'dml'], default='auto')
    ap.add_argument('--seed', type=int, default=0, help='seed of the train/val split used in training')
    ap.add_argument('--val-fraction', type=float, default=0.20)
    ap.add_argument('--per-class', action='store_true', help='also print per-class accuracy')
    ap.add_argument('--topk', type=int, default=3, help='number of predictions to print for --image')
    args = ap.parse_args()

    device = pick_device(args.device)
    ck = load_checkpoint(args.checkpoint)
    state = ck[f'{args.weights}_state']
    model, img_size, family = build_model(ck, state, args.checkpoint, args.img_size)
    model = model.to(device).eval()
    n_params = sum(p.numel() for n, p in model.named_parameters() if not n.startswith(('dec_', 'mask_token')))
    print(f'Checkpoint : {args.checkpoint} ({args.weights} weights)')
    print(f'Model      : {family}, {n_params:,} deployed parameters, input {img_size}x{img_size}, device {device}')

    tf = make_transform(img_size)
    num_classes = (state['head.weight'] if family == 'ViT' else state['classifier.3.weight']).shape[0]

    # ---- single image ----
    if args.image:
        names = class_names_from_dir(args.train_dir) or class_names_from_dir(args.data_dir)
        x = tf(Image.open(args.image).convert('RGB')).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = model(x).float().cpu().softmax(dim=1)[0]
        top = probs.topk(min(args.topk, num_classes))
        print(f'\nPredictions for {args.image}:')
        for p, idx in zip(top.values.tolist(), top.indices.tolist()):
            label = names[idx] if names and idx < len(names) else f'class {idx}'
            print(f'  {label:<24s} {p:.3f}')
        return

    # ---- dataset ----
    if args.split == 'test':
        ds = datasets.ImageFolder(args.data_dir, transform=tf)
        class_names = ds.classes
    else:
        base = datasets.ImageFolder(args.train_dir)                       # PIL images, no transform
        class_names = base.classes
        n = len(base)
        val_size = int(round(n * args.val_fraction))
        gen = torch.Generator().manual_seed(args.seed)
        _, val_subset = random_split(range(n), [n - val_size, val_size], generator=gen)
        ds = ValSplit(base, list(val_subset.indices), img_size, tf)
    if len(class_names) != num_classes:
        sys.exit(f'Dataset has {len(class_names)} classes but the checkpoint predicts {num_classes}')
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    acc, loss, correct, total, pc_correct, pc_total = run_eval(model, loader, device, num_classes)
    print(f'\n{args.split.capitalize()} accuracy: {acc:.4f}  ({correct}/{total})   cross-entropy {loss:.4f}')

    hist = ck.get('history', {}).get('val_acc') if isinstance(ck.get('history'), dict) else None
    if hist:
        logged = max(hist) if args.weights == 'best' else hist[-1]
        note = ''
        if args.split == 'val':
            note = '  <- should match the recomputed value above' if abs(logged - acc) < 1e-4 else \
                   '  <- MISMATCH: the split or preprocessing differs from training'
        print(f'Validation accuracy logged during training ({args.weights}): {logged:.4f}{note}')

    if args.per_class:
        print('\nPer-class accuracy:')
        for name, c, t in zip(class_names, pc_correct, pc_total):
            print(f'  {name:<24s} {c:>3d}/{t:<3d} {c / max(t, 1):.3f}')


if __name__ == '__main__':
    main()
