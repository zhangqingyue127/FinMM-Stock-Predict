import argparse
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
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


class CompositeDataset(Dataset):
    def __init__(self, manifest_csv, split, image_size):
        df = pd.read_csv(manifest_csv)
        self.df = df[df["split"] == split].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"No rows for split={split}")
        self.tile_size = image_size // 2
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.df)

    def make_composite(self, row):
        tile = self.tile_size
        canvas = Image.new("RGB", (tile * 2, tile * 2), "white")
        positions = [(0, 0), (tile, 0), (0, tile), (tile, tile)]
        for view, pos in zip(VIEWS, positions):
            img = Image.open(row[f"{view}_path"]).convert("RGB").resize((tile, tile), Image.BILINEAR)
            canvas.paste(img, pos)
        return canvas

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = self.make_composite(row)
        return self.transform(image), int(row["label"]), row["sample_id"]


def build_model(num_classes=2, pretrained=True):
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def eval_model(model, loader, device, criterion):
    model.eval()
    y_true, y_pred, y_prob, ids = [], [], [], []
    total_loss, total = 0.0, 0
    feats = []
    feature_hook = {}

    def hook(_, __, output):
        feature_hook["value"] = torch.flatten(output, 1).detach()

    h = model.avgpool.register_forward_hook(hook)
    with torch.no_grad():
        for x, y, sid in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            prob = torch.softmax(logits, dim=1)[:, 1]
            pred = logits.argmax(1)
            total_loss += loss.item() * x.size(0)
            total += x.size(0)
            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())
            y_prob.extend(prob.cpu().numpy().tolist())
            ids.extend(list(sid))
            feats.append(feature_hook["value"].cpu().numpy())
    h.remove()
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "loss": total_loss / max(total, 1),
        "acc": float((y_true == y_pred).mean()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": np.array(y_prob),
        "ids": ids,
        "features": np.concatenate(feats, axis=0) if feats else np.zeros((0, 512)),
    }


def retrieval_metrics(features, labels, ks=(1, 5, 10)):
    x = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    sim = x @ x.T
    np.fill_diagonal(sim, -np.inf)
    order = np.argsort(-sim, axis=1)
    return {f"retrieval_hit_at_{k}": float((labels[order[:, :k]] == labels[:, None]).any(axis=1).mean()) for k in ks}


def save_gradcam(model, dataset, output_dir, device, max_samples=4):
    ensure_dir(output_dir)
    model.eval()
    target_layer = model.layer4[-1]
    activations, gradients = {}, {}

    def fwd_hook(_, __, out):
        activations["value"] = out.detach()

    def bwd_hook(_, __, grad_out):
        gradients["value"] = grad_out[0].detach()

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)
    inv_mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    inv_std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    for idx in range(min(max_samples, len(dataset))):
        x, y, sid = dataset[idx]
        model.zero_grad(set_to_none=True)
        logits = model(x.unsqueeze(0).to(device))
        cls = int(logits.argmax(1).item())
        logits[0, cls].backward()
        act = activations["value"]
        grad = gradients["value"]
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * act).sum(dim=1).relu()
        cam = torch.nn.functional.interpolate(cam.unsqueeze(1), size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / max(cam.max() - cam.min(), 1e-12)
        img = (x.cpu() * inv_std + inv_mean).clamp(0, 1).permute(1, 2, 0).numpy()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img)
        ax.imshow(cam, cmap="jet", alpha=0.42)
        ax.set_title(f"{sid} true={int(y)} pred={cls}")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(Path(output_dir) / f"gradcam_composite_{sid}.png", dpi=180)
        plt.close(fig)
    h1.remove()
    h2.remove()


def main():
    parser = argparse.ArgumentParser(description="Train a 2x2 composite pure-image multiview ResNet.")
    parser.add_argument("--manifest_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_imagenet_init", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    out = Path(args.output_dir)
    ensure_dir(out / "checkpoints")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = device.type == "cuda"
    train_ds = CompositeDataset(args.manifest_csv, "train", args.image_size)
    val_ds = CompositeDataset(args.manifest_csv, "val", args.image_size)
    test_ds = CompositeDataset(args.manifest_csv, "test", args.image_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=(device.type == "cuda"), persistent_workers=(args.num_workers > 0))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=(device.type == "cuda"))
    model = build_model(pretrained=not args.no_imagenet_init).to(device)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_loss, best_epoch, bad = float("inf"), 0, 0
    history = []
    started = time.time()
    print(f">>> device={device} train/val/test={len(train_ds)}/{len(val_ds)}/{len(test_ds)}", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total, correct = 0.0, 0, 0
        for x, y, _ in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
            total += x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
        val = eval_model(model, val_loader, device, criterion)
        row = {"epoch": epoch, "train_loss": total_loss / max(total, 1), "train_acc": correct / max(total, 1),
               "val_loss": val["loss"], "val_acc": val["acc"], "val_f1": val["f1"]}
        history.append(row)
        print(f"[Epoch {epoch:03d}/{args.epochs:03d}] train_loss={row['train_loss']:.4f} train_acc={row['train_acc']:.4f} "
              f"val_loss={row['val_loss']:.4f} val_acc={row['val_acc']:.4f} val_f1={row['val_f1']:.4f}", flush=True)
        if val["loss"] < best_loss:
            best_loss, best_epoch, bad = val["loss"], epoch, 0
            torch.save({"model": model.state_dict(), "args": vars(args)}, out / "checkpoints" / "best_model.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print(">>> Early stopping triggered.", flush=True)
                break

    model.load_state_dict(torch.load(out / "checkpoints" / "best_model.pt", map_location=device)["model"])
    test = eval_model(model, test_loader, device, criterion)
    metrics = {
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "test_loss": test["loss"],
        "test_acc": test["acc"],
        "test_precision_weighted": test["precision"],
        "test_recall_weighted": test["recall"],
        "test_f1_weighted": test["f1"],
        "confusion_matrix": confusion_matrix(test["y_true"], test["y_pred"]).tolist(),
        "retrieval": retrieval_metrics(test["features"], test["y_true"]),
        "classification_report": classification_report(test["y_true"], test["y_pred"], output_dict=True, zero_division=0),
        "args": vars(args),
        "total_time_min": (time.time() - started) / 60.0,
    }
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    save_json(metrics, out / "test_metrics.json")
    save_gradcam(model, test_ds, out / "gradcam", device=device, max_samples=4)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
