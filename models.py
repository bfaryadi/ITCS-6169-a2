"""Model definitions for ITCS 6169/8169 Assignment 2 (self-supervised representation learning).

Everything here is trained from scratch on the 2,400 provided images (no pretrained weights).

Contents
--------
CNN family (Assignment 1 architecture and the SSL wrappers built on it)
    TNet               - the A1 CNN: conv stack -> global average pool -> dropout -> Linear(256, 16).
                         This is also the deployed architecture of the SigLIP-style and BYOL models after
                         fine-tuning (the SSL heads are discarded).
    SSLNet             - TNet conv stack + global pool + projection head (used only during SSL pretraining).
    SigLIPStyleLoss    - pairwise sigmoid loss with a learnable temperature and bias.
    build_byol_predictor, byol_loss, ema_update - the BYOL-specific pieces (predictor MLP, loss, EMA).

ViT family (supervised ViT and ViT + SSAT share one class)
    ViTSSAT            - ViT encoder + classification head (primary stream) and a small decoder that
                         reconstructs hidden patches (auxiliary stream, training only). The supervised ViT is
                         the same model trained with lambda = 1 (auxiliary loss switched off).
    vit_kwargs         - picks the architecture arguments out of a saved training config.

The class bodies are copied from the training notebooks; only notebook-level globals (image size, number
of classes, feature width) were turned into constructor arguments.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------------------------
# CNN family
# --------------------------------------------------------------------------------------------------
def conv_bn_relu(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class TNet(nn.Module):
    def __init__(self, num_classes=16, widths=(32, 64, 128, 256), dropout=0.3):
        super().__init__()
        c1, c2, c3, c4 = widths
        self.features = nn.Sequential(
            conv_bn_relu(3, c1), conv_bn_relu(c1, c1), nn.MaxPool2d(2),
            conv_bn_relu(c1, c2), conv_bn_relu(c2, c2), nn.MaxPool2d(2),
            conv_bn_relu(c2, c3), conv_bn_relu(c3, c3), nn.MaxPool2d(2),
            conv_bn_relu(c3, c4), conv_bn_relu(c4, c4),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c4, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class SSLNet(nn.Module):
    """Encoder (TNet conv stack) + global pool + projection head."""
    def __init__(self, feat_dim=256, hidden=512, proj_dim=128, num_classes=16):
        super().__init__()
        self.features = TNet(num_classes=num_classes).features
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        self.head = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, proj_dim),
        )

    def embed(self, x):          # representation used for probing / fine-tuning
        return self.pool(self.features(x))

    def forward(self, x):        # projection used by the SSL loss
        return self.head(self.embed(x))


class SigLIPStyleLoss(nn.Module):
    """Pairwise sigmoid loss: +1 label for matching pairs, -1 for every other pair in the batch."""
    def __init__(self, init_t=10.0, init_b=-10.0):
        super().__init__()
        self.log_t = nn.Parameter(torch.tensor(math.log(init_t)))
        self.b = nn.Parameter(torch.tensor(float(init_b)))

    def forward(self, z1, z2):
        z1, z2 = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
        logits = self.log_t.exp() * (z1 @ z2.T) + self.b
        labels = (2 * torch.eye(len(z1)) - 1).to(z1.device)   # build on CPU: torch.eye is broken on DirectML
        return -F.logsigmoid(labels * logits).sum() / len(z1)


# ---- BYOL pieces ----
def build_byol_predictor(proj_dim=128, pred_hidden=512):
    """Predictor MLP on top of the online projection (online side only; the target has no predictor)."""
    return nn.Sequential(
        nn.Linear(proj_dim, pred_hidden),
        nn.BatchNorm1d(pred_hidden),
        nn.ReLU(inplace=True),
        nn.Linear(pred_hidden, proj_dim),
    )


def byol_loss(p, z):
    """2 - 2*cos(p, z) with stop-gradient on the target side."""
    p = F.normalize(p, dim=1)
    z = F.normalize(z.detach(), dim=1)
    return (2 - 2 * (p * z).sum(dim=1)).mean()


@torch.no_grad()
def ema_update(target, online, tau):
    """target <- tau*target + (1-tau)*online. Parameters only: the target always runs in train mode, so its BN
    running statistics are never used. mul_/add_(alpha) avoids a temporary (and lerp falls back to CPU on DirectML)."""
    for pt, po in zip(target.parameters(), online.parameters()):
        pt.mul_(tau).add_(po, alpha=1.0 - tau)


# --------------------------------------------------------------------------------------------------
# ViT family
# --------------------------------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, dim, heads, drop=0.0):
        super().__init__()
        self.heads, self.scale = heads, (dim // heads) ** -0.5
        self.qkv, self.proj, self.drop = nn.Linear(dim, dim * 3), nn.Linear(dim, dim), nn.Dropout(drop)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = [t.reshape(B, N, self.heads, C // self.heads).transpose(1, 2) for t in self.qkv(x).chunk(3, dim=-1)]
        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        return self.drop(self.proj((attn @ v).transpose(1, 2).reshape(B, N, C)))


class Block(nn.Module):
    def __init__(self, dim, heads, drop=0.0, mlp_ratio=4):
        super().__init__()
        self.n1, self.attn, self.n2 = nn.LayerNorm(dim), Attention(dim, heads, drop), nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(dim * mlp_ratio, dim), nn.Dropout(drop))

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))


class ViTSSAT(nn.Module):
    def __init__(self, img=64, patch=8, dim=192, depth=6, heads=3, dec_dim=128, dec_depth=2, dec_heads=4,
                 drop=0.1, num_classes=16):
        super().__init__()
        self.N, self.patch = (img // patch) ** 2, patch
        # shared encoder
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.N + 1, dim))
        self.blocks = nn.ModuleList([Block(dim, heads, drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)                      # primary stream
        # auxiliary stream (training only)
        self.dec_embed = nn.Linear(dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.dec_pos = nn.Parameter(torch.zeros(1, self.N, dec_dim))
        self.dec_blocks = nn.ModuleList([Block(dec_dim, dec_heads, 0.0) for _ in range(dec_depth)])
        self.dec_norm = nn.LayerNorm(dec_dim)
        self.dec_pred = nn.Linear(dec_dim, patch * patch * 3)
        for q in (self.cls_token, self.pos_embed, self.dec_pos):
            nn.init.trunc_normal_(q, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    def _tokens(self, x):
        return self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos_embed[:, 1:]

    def _encode(self, t):
        z = torch.cat([(self.cls_token + self.pos_embed[:, :1]).expand(t.size(0), -1, -1), t], dim=1)
        for b in self.blocks:
            z = b(z)
        return self.norm(z)

    def forward(self, x):                                            # primary stream: class logits
        return self.head(self._encode(self._tokens(x))[:, 0])

    def aux_loss(self, xa, target, sel, unshuf, mask):               # auxiliary stream: reconstruct hidden patches
        vis = self._encode(sel @ self._tokens(xa))[:, 1:]            # encoder sees only the visible patches
        d = self.dec_embed(vis)
        B, K, _ = d.shape
        full = unshuf @ torch.cat([d, self.mask_token.expand(B, self.N - K, -1)], dim=1) + self.dec_pos
        for b in self.dec_blocks:
            full = b(full)
        pred = self.dec_pred(self.dec_norm(full))
        return (((pred - target) ** 2).mean(-1) * mask).sum() / mask.sum()      # MSE on hidden patches only


def vit_kwargs(cfg):
    """Architecture arguments of ViTSSAT taken from a saved training config (ckpt['hparams']['cfg'])."""
    return {k: cfg[k] for k in ('patch', 'dim', 'depth', 'heads', 'dec_dim', 'dec_depth', 'dec_heads', 'drop')}


if __name__ == '__main__':
    def n_params(m, skip=()):
        return sum(p.numel() for n, p in m.named_parameters() if not n.startswith(skip))

    print(f'TNet (A1 CNN)                : {n_params(TNet()):,} parameters')
    print(f'SSLNet (with projection head): {n_params(SSLNet()):,} parameters')
    for img, patch in ((64, 8), (128, 16)):
        m = ViTSSAT(img=img, patch=patch)
        print(f'ViT {img}px (patch {patch}): encoder+classifier {n_params(m, ("dec_", "mask_token")):,} | '
              f'decoder (training only) {n_params(m) - n_params(m, ("dec_", "mask_token")):,}')
