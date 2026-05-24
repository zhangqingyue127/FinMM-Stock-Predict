import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights


VIEWS = ["candle", "ma", "rsi", "macd"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MultiViewPairDataset(Dataset):
    def __init__(self, manifest_csv, split, image_size):
        df = pd.read_csv(manifest_csv)
        self.df = df[df["split"] == split].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"No rows for split={split}")
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomApply([transforms.ColorJitter(0.12, 0.12, 0.04, 0.0)], p=0.4),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        v1, v2 = random.sample(VIEWS, 2)
        x1 = self.transform(Image.open(row[f"{v1}_path"]).convert("RGB"))
        x2 = self.transform(Image.open(row[f"{v2}_path"]).convert("RGB"))
        return x1, x2


class ContrastiveEncoder(nn.Module):
    def __init__(self, projection_dim=128, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.encoder = backbone
        self.projector = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, projection_dim),
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return nn.functional.normalize(z, dim=1)


def nt_xent(z1, z2, temperature):
    n = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    logits = torch.matmul(z, z.T) / temperature
    logits = logits.masked_fill(torch.eye(2 * n, device=z.device, dtype=torch.bool), torch.finfo(logits.dtype).min)
    target = torch.cat([torch.arange(n, 2 * n, device=z.device), torch.arange(0, n, device=z.device)])
    return nn.functional.cross_entropy(logits, target)


def main():
    parser = argparse.ArgumentParser(description="Multi-view consistency pretraining for pure candlestick images.")
    parser.add_argument("--manifest_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--projection_dim", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_imagenet_init", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    out = Path(args.output_dir)
    ensure_dir(out)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = device.type == "cuda"

    ds = MultiViewPairDataset(args.manifest_csv, split="train", image_size=args.image_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"), drop_last=True,
                        persistent_workers=(args.num_workers > 0))
    model = ContrastiveEncoder(args.projection_dim, pretrained=not args.no_imagenet_init).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    history = []
    best = float("inf")
    started = time.time()
    print(f">>> device={device} train_samples={len(ds)}", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total, seen = 0.0, 0
        ep_start = time.time()
        for x1, x2 in loader:
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                loss = nt_xent(model(x1), model(x2), args.temperature)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * x1.size(0)
            seen += x1.size(0)
        avg = total / max(seen, 1)
        history.append({"epoch": epoch, "loss": avg, "epoch_time_sec": time.time() - ep_start})
        ckpt = {"encoder_state_dict": model.encoder.state_dict(), "projector_state_dict": model.projector.state_dict(),
                "args": vars(args), "epoch": epoch, "loss": avg}
        torch.save(ckpt, out / "last_multiview_encoder.pt")
        if avg < best:
            best = avg
            torch.save(ckpt, out / "best_multiview_encoder.pt")
        print(f"[Epoch {epoch:03d}/{args.epochs:03d}] loss={avg:.5f} time={history[-1]['epoch_time_sec']:.1f}s", flush=True)

    save_json({"args": vars(args), "best_loss": best, "history": history,
               "total_time_min": (time.time() - started) / 60.0}, out / "pretrain_summary.json")
    print(">>> best checkpoint:", out / "best_multiview_encoder.pt", flush=True)


if __name__ == "__main__":
    main()
