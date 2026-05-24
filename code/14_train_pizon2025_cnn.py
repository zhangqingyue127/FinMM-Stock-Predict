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
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


class PizonCNN(nn.Module):
    def __init__(self, weight_decay=1e-4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01, inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01, inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01, inplace=True),
            nn.MaxPool2d(2),
        )
        self.dropout1 = nn.Dropout(0.5)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 12 * 12, 128),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.dropout1(x)
        return self.classifier(x)


def eval_model(model, loader, device, criterion):
    model.eval()
    total_loss, total = 0.0, 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            total += x.size(0)
            pred = logits.argmax(1)
            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = float((y_true == y_pred).mean())
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "loss": total_loss / max(total, 1),
        "acc": acc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def plot_history(history, out_dir):
    df = pd.DataFrame(history)
    df.to_csv(out_dir / "history.csv", index=False)
    for metric in ["loss", "acc"]:
        plt.figure(figsize=(8, 5))
        plt.plot(df["epoch"], df[f"train_{metric}"], label=f"train_{metric}")
        plt.plot(df["epoch"], df[f"val_{metric}"], label=f"val_{metric}")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{metric}_curve.png", dpi=200)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Reproduce Pizon et al. 2025 lightweight CNN for candlestick trend classification.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    out = Path(args.output_dir)
    ensure_dir(out / "checkpoints")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = (device.type == "cuda")

    tfm_train = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.98, 1.02)),
        transforms.ToTensor(),
    ])
    tfm_eval = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
    ])
    data_root = Path(args.data_root)
    train_ds = datasets.ImageFolder(data_root / "train", transform=tfm_train)
    val_ds = datasets.ImageFolder(data_root / "val", transform=tfm_eval)
    test_ds = datasets.ImageFolder(data_root / "test", transform=tfm_eval)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=(device.type == "cuda"), persistent_workers=(args.num_workers > 0))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=(device.type == "cuda"))

    model = PizonCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    best_val_loss = float("inf")
    best_epoch = 0
    bad = 0
    history = []
    start = time.time()

    print(f">>> device: {device}")
    print(f">>> class_to_idx: {train_ds.class_to_idx}")
    print(f">>> train/val/test: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total, correct = 0.0, 0, 0
        ep_start = time.time()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            total += x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
        train_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        val = eval_model(model, val_loader, device, criterion)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val["loss"],
            "val_acc": val["acc"],
            "val_f1": val["f1"],
            "epoch_time_sec": time.time() - ep_start,
        }
        history.append(row)
        print(
            f"[Epoch {epoch:03d}/{args.epochs:03d}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val['loss']:.4f} val_acc={val['acc']:.4f} val_f1={val['f1']:.4f} "
            f"time={row['epoch_time_sec']:.1f}s",
            flush=True,
        )
        if val["loss"] < best_val_loss:
            best_val_loss = val["loss"]
            best_epoch = epoch
            bad = 0
            torch.save({"model": model.state_dict(), "args": vars(args), "class_to_idx": train_ds.class_to_idx},
                       out / "checkpoints" / "best_model.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print(">>> Early stopping triggered.", flush=True)
                break

    ckpt = torch.load(out / "checkpoints" / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    test = eval_model(model, test_loader, device, criterion)
    cm = confusion_matrix(test["y_true"], test["y_pred"])
    report = classification_report(test["y_true"], test["y_pred"], target_names=test_ds.classes, output_dict=True, zero_division=0)
    metrics = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_loss": test["loss"],
        "test_acc": test["acc"],
        "test_precision_weighted": test["precision"],
        "test_recall_weighted": test["recall"],
        "test_f1_weighted": test["f1"],
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "class_to_idx": train_ds.class_to_idx,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "total_time_min": (time.time() - start) / 60.0,
        "args": vars(args),
    }
    save_json(metrics, out / "test_metrics.json")
    plot_history(history, out)
    with open(out / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(test["y_true"], test["y_pred"], target_names=test_ds.classes, zero_division=0))
    print("\n>>> Test Results")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
