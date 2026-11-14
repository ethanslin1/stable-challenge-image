# """LeJEPA (SIGReg + invariance) on the challenge training set, then export to ONNX.

#     uv run train_lejepa.py                      # trains, writes model.onnx
#     uv run train_lejepa.py --epochs 400 --batch 256 --out model.onnx

# Put this file next to training_data.py, export_onnx.py and evaluate.py.

# Loss (Balestriero & LeCun, "LeJEPA"):
#     L = (1 - lam) * invariance + lam * SIGReg
#     invariance: mean squared distance of every view's projection to the mean projection of the global views
#     SIGReg:     Epps-Pulley characteristic-function test of the projections against N(0, I), averaged over
#                 random 1-D slices (no stop-gradient, no teacher, no predictor, no EMA)

# Labels are NOT used in the loss. They are only used to print a monitoring score (the same ridge probe that
# evaluate.py uses) on the labelled 10%.
# """

# import argparse
# import math
# import time

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, Dataset
# from torchvision.transforms import v2

# from evaluate import Ridge
# from export_onnx import MEAN, STD, export_onnx
# from training_data import UNLABELLED, load_training_data


# # ------------------------------------------------------------------------------------------ encoder
# class Block(nn.Module):
#     def __init__(self, cin, cout, stride):
#         super().__init__()
#         self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
#         self.b1 = nn.BatchNorm2d(cout)
#         self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
#         self.b2 = nn.BatchNorm2d(cout)
#         self.short = None
#         if stride != 1 or cin != cout:
#             self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

#     def forward(self, x):
#         y = self.b2(self.c2(F.relu(self.b1(self.c1(x)))))
#         return F.relu(y + (x if self.short is None else self.short(x)))


# class Encoder(nn.Module):
#     """Small ResNet-18-style encoder. widths (40, 80, 160, 320) -> ~4.4M params (~17.6 MB float32, cap is 20 MiB).
#     Output: [B, widths[-1]] (zero-padded to 1024 by export_onnx)."""

#     def __init__(self, widths=(40, 80, 160, 320), depth=(2, 2, 2, 2)):
#         super().__init__()
#         self.stem = nn.Sequential(nn.Conv2d(3, widths[0], 3, 1, 1, bias=False), nn.BatchNorm2d(widths[0]),
#                                   nn.ReLU(inplace=True))
#         layers, cin = [], widths[0]
#         for i, (w, d) in enumerate(zip(widths, depth)):
#             for j in range(d):
#                 layers.append(Block(cin, w, 2 if (j == 0 and i > 0) else 1))
#                 cin = w
#         self.layers = nn.Sequential(*layers)
#         self.dim = widths[-1]

#     def forward(self, x):
#         return self.layers(self.stem(x)).mean((2, 3))


# class Projector(nn.Module):
#     def __init__(self, din, hidden=1024, dout=128):
#         super().__init__()
#         self.net = nn.Sequential(nn.Linear(din, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
#                                  nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
#                                  nn.Linear(hidden, dout))

#     def forward(self, x):
#         return self.net(x)


# # ------------------------------------------------------------------------------------------ SIGReg
# class SIGReg(nn.Module):
#     """Sketched Isotropic Gaussian Regularisation (Epps-Pulley statistic on random slices)."""

#     def __init__(self, knots=17, t_max=3.0, slices=256):
#         super().__init__()
#         t = torch.linspace(0, t_max, knots)
#         dt = t_max / (knots - 1)
#         w = torch.full((knots,), 2 * dt)
#         w[[0, -1]] = dt
#         phi = torch.exp(-t.square() / 2.0)  # characteristic function of N(0, 1)
#         self.slices = slices
#         self.register_buffer("t", t)
#         self.register_buffer("phi", phi)
#         self.register_buffer("w", w * phi)

#     def forward(self, proj):  # proj: [V, N, K]
#         A = torch.randn(proj.size(-1), self.slices, device=proj.device, dtype=proj.dtype)
#         A = A / A.norm(dim=0, keepdim=True)
#         x_t = (proj @ A).unsqueeze(-1) * self.t  # [V, N, S, T]
#         err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()  # mean over N
#         return ((err @ self.w) * proj.size(-2)).mean()


# # ------------------------------------------------------------------------------------------ data
# class Views(Dataset):
#     """Returns n_global views at 96px and n_local views at `local_size` px of one image."""

#     def __init__(self, images, n_global=2, n_local=4, local_size=48):
#         self.images, self.ng, self.nl = images, n_global, n_local
#         norm = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)]

#         def aug(size, scale, blur):
#             t = [v2.RandomResizedCrop(size, scale=scale, antialias=True),
#                  v2.RandomHorizontalFlip(), v2.RandomVerticalFlip(),  # data includes aerial/galaxy images
#                  v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.2, 0.05)], p=0.8),
#                  v2.RandomGrayscale(0.2)]
#             if blur:
#                 t.append(v2.RandomApply([v2.GaussianBlur(5, (0.1, 1.5))], p=0.3))
#             return v2.Compose(t + norm)

#         self.g = aug(96, (0.4, 1.0), True)
#         self.l = aug(local_size, (0.1, 0.4), False)

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, i):
#         im = self.images[i]
#         return (torch.stack([self.g(im) for _ in range(self.ng)]),
#                 torch.stack([self.l(im) for _ in range(self.nl)]))


# # ------------------------------------------------------------------------------------------ monitoring
# @torch.no_grad()
# def probe_score(enc, imgs, labels, dev):
#     """Leave-one-out ridge-probe accuracy on the labelled images (monitoring only, not in the loss)."""
#     enc.eval()
#     t = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)])
#     x = torch.stack([t(im) for im in imgs])
#     z = torch.cat([enc(x[i:i + 256].to(dev)).float().cpu() for i in range(0, len(x), 256)]).numpy()
#     enc.train()
#     return float((Ridge().fit(z, labels).loo_pred == labels).mean())


# # ------------------------------------------------------------------------------------------ main
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--epochs", type=int, default=300)
#     ap.add_argument("--batch", type=int, default=256)
#     ap.add_argument("--lr", type=float, default=2e-3)
#     ap.add_argument("--wd", type=float, default=0.05)
#     ap.add_argument("--lam", type=float, default=0.05, help="SIGReg weight")
#     ap.add_argument("--n-global", type=int, default=2)
#     ap.add_argument("--n-local", type=int, default=4)
#     ap.add_argument("--workers", type=int, default=8)
#     ap.add_argument("--eval-every", type=int, default=10)
#     ap.add_argument("--out", default="model.onnx")
#     ap.add_argument("--seed", type=int, default=0)
#     a = ap.parse_args()

#     torch.manual_seed(a.seed)
#     np.random.seed(a.seed)
#     dev = "cuda" if torch.cuda.is_available() else "cpu"
#     amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda")

#     ds = load_training_data()
#     images = [im.convert("RGB") for im in ds["image"]]
#     labels = np.asarray(ds["label"])
#     lab_idx = np.flatnonzero(labels != UNLABELLED)
#     lab_imgs, lab_y = [images[i] for i in lab_idx], labels[lab_idx]
#     print(f"{len(images)} images ({len(lab_idx)} labelled, used for monitoring only), device {dev}")

#     enc = Encoder().to(dev)
#     proj = Projector(enc.dim).to(dev)
#     n_params = sum(p.numel() for p in enc.parameters())
#     print(f"encoder params {n_params / 1e6:.2f}M (~{n_params * 4 / 2**20:.1f} MiB float32; cap 20 MiB)")
#     assert n_params * 4 < 19 * 2**20, "encoder too big for the 20 MiB cap; reduce widths"
#     sigreg = SIGReg().to(dev)

#     loader = DataLoader(Views(images, a.n_global, a.n_local), batch_size=a.batch, shuffle=True, drop_last=True,
#                         num_workers=a.workers, pin_memory=dev == "cuda", persistent_workers=a.workers > 0)
#     params = list(enc.parameters()) + list(proj.parameters())
#     decay = [p for p in params if p.ndim > 1]
#     no_decay = [p for p in params if p.ndim <= 1]
#     opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": no_decay, "weight_decay": 0}],
#                             lr=a.lr, betas=(0.9, 0.95))
#     total, warm = a.epochs * len(loader), 10 * len(loader)
#     sched = torch.optim.lr_scheduler.LambdaLR(
#         opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (total - warm))))

#     enc.train()
#     t0 = time.time()
#     for ep in range(a.epochs):
#         tot = inv_tot = sig_tot = 0.0
#         for g, l in loader:
#             B = g.size(0)
#             g = g.to(dev, non_blocking=True).transpose(0, 1).reshape(-1, *g.shape[2:])  # [Vg*B, ...]
#             l = l.to(dev, non_blocking=True).transpose(0, 1).reshape(-1, *l.shape[2:])  # [Vl*B, ...]
#             with torch.autocast(**amp):
#                 pg = proj(enc(g)).float().view(a.n_global, B, -1)
#                 pl = proj(enc(l)).float().view(a.n_local, B, -1)
#             p = torch.cat([pg, pl])  # [V, B, K]
#             inv = (pg.mean(0, keepdim=True) - p).square().mean()
#             sig = sigreg(p)
#             loss = (1 - a.lam) * inv + a.lam * sig
#             opt.zero_grad(set_to_none=True)
#             loss.backward()
#             opt.step()
#             sched.step()
#             tot, inv_tot, sig_tot = tot + loss.item(), inv_tot + inv.item(), sig_tot + sig.item()
#         n = len(loader)
#         msg = f"ep {ep + 1:4d}/{a.epochs} loss {tot / n:.4f} inv {inv_tot / n:.4f} sigreg {sig_tot / n:.4f} " \
#               f"lr {sched.get_last_lr()[0]:.2e} [{time.time() - t0:.0f}s]"
#         if (ep + 1) % a.eval_every == 0 or ep + 1 == a.epochs:
#             msg += f"  probe LOO acc {100 * probe_score(enc, lab_imgs, lab_y, dev):.1f}"
#         print(msg, flush=True)

#     enc.eval().cpu()
#     export_onnx(enc, a.out)
#     print(f"wrote {a.out}. Next: uv run test_submission.py {a.out}")


# if __name__ == "__main__":
#     main()

"""LeJEPA (SIGReg + invariance) on the challenge training set, then export to ONNX.

    uv run train_lejepa.py                      # trains, writes model.onnx
    uv run train_lejepa.py --extended --epochs 60   # broader data (run extra_data.py first, or it auto-builds)
    uv run train_lejepa.py --epochs 400 --batch 256 --out model.onnx

Put this file next to training_data.py, export_onnx.py and evaluate.py.

Loss (Balestriero & LeCun, "LeJEPA"):
    L = (1 - lam) * invariance + lam * SIGReg
    invariance: mean squared distance of every view's projection to the mean projection of the global views
    SIGReg:     Epps-Pulley characteristic-function test of the projections against N(0, I), averaged over
                random 1-D slices (no stop-gradient, no teacher, no predictor, no EMA)

Labels are NOT used in the loss. They are only used to print a monitoring score (the same ridge probe that
evaluate.py uses) on the labelled 10%.
"""

import argparse
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from evaluate import Ridge
from export_onnx import MEAN, STD, export_onnx
from extra_data import load_extended_data
from training_data import UNLABELLED, load_training_data


# ------------------------------------------------------------------------------------------ encoder
class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = self.b2(self.c2(F.relu(self.b1(self.c1(x)))))
        return F.relu(y + (x if self.short is None else self.short(x)))


class Encoder(nn.Module):
    """Small ResNet-18-style encoder. widths (40, 80, 160, 320) -> ~4.4M params (~17.6 MB float32, cap is 20 MiB).
    Output: [B, widths[-1]] (zero-padded to 1024 by export_onnx)."""

    def __init__(self, widths=(40, 80, 160, 320), depth=(2, 2, 2, 2)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, widths[0], 3, 1, 1, bias=False), nn.BatchNorm2d(widths[0]),
                                  nn.ReLU(inplace=True))
        layers, cin = [], widths[0]
        for i, (w, d) in enumerate(zip(widths, depth)):
            for j in range(d):
                layers.append(Block(cin, w, 2 if (j == 0 and i > 0) else 1))
                cin = w
        self.layers = nn.Sequential(*layers)
        self.dim = widths[-1]

    def forward(self, x):
        return self.layers(self.stem(x)).mean((2, 3))


class Projector(nn.Module):
    def __init__(self, din, hidden=1024, dout=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
                                 nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
                                 nn.Linear(hidden, dout))

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------------------------------------ SIGReg
class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularisation (Epps-Pulley statistic on random slices)."""

    def __init__(self, knots=17, t_max=3.0, slices=256):
        super().__init__()
        t = torch.linspace(0, t_max, knots)
        dt = t_max / (knots - 1)
        w = torch.full((knots,), 2 * dt)
        w[[0, -1]] = dt
        phi = torch.exp(-t.square() / 2.0)  # characteristic function of N(0, 1)
        self.slices = slices
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("w", w * phi)

    def forward(self, proj):  # proj: [V, N, K]
        A = torch.randn(proj.size(-1), self.slices, device=proj.device, dtype=proj.dtype)
        A = A / A.norm(dim=0, keepdim=True)
        x_t = (proj @ A).unsqueeze(-1) * self.t  # [V, N, S, T]
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()  # mean over N
        return ((err @ self.w) * proj.size(-2)).mean()


# ------------------------------------------------------------------------------------------ data
class Views(Dataset):
    """Returns n_global views at 96px and n_local views at `local_size` px of one image."""

    def __init__(self, images, n_global=2, n_local=4, local_size=48):
        self.images, self.ng, self.nl = images, n_global, n_local
        norm = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)]

        def aug(size, scale, blur):
            t = [v2.RandomResizedCrop(size, scale=scale, antialias=True),
                 v2.RandomHorizontalFlip(), v2.RandomVerticalFlip(),  # data includes aerial/galaxy images
                 v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.2, 0.05)], p=0.8),
                 v2.RandomGrayscale(0.2)]
            if blur:
                t.append(v2.RandomApply([v2.GaussianBlur(5, (0.1, 1.5))], p=0.3))
            return v2.Compose(t + norm)

        self.g = aug(96, (0.4, 1.0), True)
        self.l = aug(local_size, (0.1, 0.4), False)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        im = self.images[i]
        return (torch.stack([self.g(im) for _ in range(self.ng)]),
                torch.stack([self.l(im) for _ in range(self.nl)]))


# ------------------------------------------------------------------------------------------ monitoring
@torch.no_grad()
def probe_score(enc, imgs, labels, dev):
    """Leave-one-out ridge-probe accuracy on the labelled images (monitoring only, not in the loss)."""
    enc.eval()
    t = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)])
    x = torch.stack([t(im) for im in imgs])
    z = torch.cat([enc(x[i:i + 256].to(dev)).float().cpu() for i in range(0, len(x), 256)]).numpy()
    enc.train()
    return float((Ridge().fit(z, labels).loo_pred == labels).mean())


# ------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--lam", type=float, default=0.05, help="SIGReg weight")
    ap.add_argument("--n-global", type=int, default=2)
    ap.add_argument("--n-local", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--extended", action="store_true",
                    help="train on data/train_extended.parquet (original + extra unlabelled images from extra_data.py)")
    ap.add_argument("--data", default=None, help="path to any parquet with image/label columns (overrides --extended)")
    ap.add_argument("--out", default="model.onnx")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda")

    if a.data:
        ds = load_training_data(a.data)
    elif a.extended:
        ds = load_extended_data()
    else:
        ds = load_training_data()
    images = [im.convert("RGB") for im in ds["image"]]
    labels = np.asarray(ds["label"])
    lab_idx = np.flatnonzero(labels != UNLABELLED)
    lab_imgs, lab_y = [images[i] for i in lab_idx], labels[lab_idx]
    print(f"{len(images)} images ({len(lab_idx)} labelled, used for monitoring only), device {dev}")

    enc = Encoder().to(dev)
    proj = Projector(enc.dim).to(dev)
    n_params = sum(p.numel() for p in enc.parameters())
    print(f"encoder params {n_params / 1e6:.2f}M (~{n_params * 4 / 2**20:.1f} MiB float32; cap 20 MiB)")
    assert n_params * 4 < 19 * 2**20, "encoder too big for the 20 MiB cap; reduce widths"
    sigreg = SIGReg().to(dev)

    loader = DataLoader(Views(images, a.n_global, a.n_local), batch_size=a.batch, shuffle=True, drop_last=True,
                        num_workers=a.workers, pin_memory=dev == "cuda", persistent_workers=a.workers > 0)
    params = list(enc.parameters()) + list(proj.parameters())
    decay = [p for p in params if p.ndim > 1]
    no_decay = [p for p in params if p.ndim <= 1]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": no_decay, "weight_decay": 0}],
                            lr=a.lr, betas=(0.9, 0.95))
    total, warm = a.epochs * len(loader), max(1, min(10, a.epochs // 10)) * len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (total - warm))))

    enc.train()
    t0 = time.time()
    for ep in range(a.epochs):
        tot = inv_tot = sig_tot = 0.0
        for g, l in loader:
            B = g.size(0)
            g = g.to(dev, non_blocking=True).transpose(0, 1).reshape(-1, *g.shape[2:])  # [Vg*B, ...]
            l = l.to(dev, non_blocking=True).transpose(0, 1).reshape(-1, *l.shape[2:])  # [Vl*B, ...]
            with torch.autocast(**amp):
                pg = proj(enc(g)).float().view(a.n_global, B, -1)
                pl = proj(enc(l)).float().view(a.n_local, B, -1)
            p = torch.cat([pg, pl])  # [V, B, K]
            inv = (pg.mean(0, keepdim=True) - p).square().mean()
            sig = sigreg(p)
            loss = (1 - a.lam) * inv + a.lam * sig
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot, inv_tot, sig_tot = tot + loss.item(), inv_tot + inv.item(), sig_tot + sig.item()
        n = len(loader)
        msg = f"ep {ep + 1:4d}/{a.epochs} loss {tot / n:.4f} inv {inv_tot / n:.4f} sigreg {sig_tot / n:.4f} " \
              f"lr {sched.get_last_lr()[0]:.2e} [{time.time() - t0:.0f}s]"
        if (ep + 1) % a.eval_every == 0 or ep + 1 == a.epochs:
            msg += f"  probe LOO acc {100 * probe_score(enc, lab_imgs, lab_y, dev):.1f}"
        print(msg, flush=True)

    enc.eval().cpu()
    export_onnx(enc, a.out)
    print(f"wrote {a.out}. Next: uv run test_submission.py {a.out}")


if __name__ == "__main__":
    main()
