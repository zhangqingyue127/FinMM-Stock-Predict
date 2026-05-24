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


DEFAULT_VIEWS = ["candle", "ma", "rsi", "macd"]
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


class MultiViewDataset(Dataset):
    def __init__(self, manifest_csv, split, image_size, views, augment=False, shuffle_labels=False, seed=42):
        df = pd.read_csv(manifest_csv)
        self.df = df[df["split"] == split].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"No rows for split={split}")
        if shuffle_labels:
            rng = np.random.default_rng(seed)
            self.df["label"] = rng.permutation(self.df["label"].to_numpy())
            print(f">>> shuffled labels for split={split}", flush=True)
        self.views = views
        ops = [transforms.Resize((image_size, image_size))]
        if augment:
            ops.extend([
                transforms.RandomAffine(degrees=0, translate=(0.025, 0.025), scale=(0.98, 1.02)),
                transforms.RandomApply([transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.02, hue=0.0)], p=0.35),
            ])
        ops.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        self.transform = transforms.Compose(ops)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        xs = [self.transform(Image.open(row[f"{v}_path"]).convert("RGB")) for v in self.views]
        return torch.stack(xs, dim=0), int(row["label"]), row["sample_id"]


class MultiViewClassifier(nn.Module):
    def __init__(
        self,
        encoder_ckpt="",
        pretrained=True,
        num_classes=2,
        freeze_encoder=False,
        fusion="mean",
        num_views=4,
        dropout=0.25,
    ):
        super().__init__()
        if fusion not in {"mean", "concat"}:
            raise ValueError(f"unknown fusion: {fusion}")
        self.fusion = fusion
        self.num_views = num_views
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        encoder = models.resnet18(weights=weights)
        dim = encoder.fc.in_features
        encoder.fc = nn.Identity()
        if encoder_ckpt:
            ckpt = torch.load(encoder_ckpt, map_location="cpu")
            state = ckpt.get("encoder_state_dict", ckpt)
            cleaned = {k.replace("module.", ""): v for k, v in state.items() if not k.startswith("fc.")}
            encoder.load_state_dict(cleaned, strict=False)
        if freeze_encoder:
            for p in encoder.parameters():
                p.requires_grad = False
        self.encoder = encoder
        if fusion == "concat":
            self.classifier = nn.Sequential(
                nn.Linear(dim * num_views, dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(dim, num_classes),
            )
        else:
            self.classifier = nn.Linear(dim, num_classes)

    def forward_features(self, x):
        b, v, c, h, w = x.shape
        feat = self.encoder(x.view(b * v, c, h, w)).view(b, v, -1)
        if self.fusion == "concat":
            return feat.reshape(b, v * feat.size(-1))
        return feat.mean(dim=1)

    def forward(self, x):
        return self.classifier(self.forward_features(x))


def eval_model(model, loader, device, criterion):
    model.eval()
    y_true, y_pred, probs, ids = [], [], [], []
    total_loss, total = 0.0, 0
    feats = []
    with torch.no_grad():
        for x, y, sid in loader:
            x = x.to(device)
            y = y.to(device)
            feat = model.forward_features(x)
            logits = model.classifier(feat)
            loss = criterion(logits, y)
            prob = torch.softmax(logits, dim=1)[:, 1]
            pred = logits.argmax(1)
            total_loss += loss.item() * x.size(0)
            total += x.size(0)
            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())
            probs.extend(prob.cpu().numpy().tolist())
            ids.extend(list(sid))
            feats.append(feat.cpu().numpy())
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "loss": total_loss / max(total, 1),
        "acc": float((y_true == y_pred).mean()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "y_true": y_true,
        "y_pred": y_pred,
        "prob_up": np.array(probs),
        "ids": ids,
        "features": np.concatenate(feats, axis=0) if feats else np.zeros((0, 512)),
    }


def retrieval_metrics(features, labels, ks=(1, 5, 10)):
    x = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    sim = x @ x.T
    np.fill_diagonal(sim, -np.inf)
    order = np.argsort(-sim, axis=1)
    out = {}
    for k in ks:
        top = order[:, :k]
        hit = (labels[top] == labels[:, None]).any(axis=1)
        out[f"retrieval_hit_at_{k}"] = float(hit.mean())
    return out


def save_gradcam(model, dataset, output_dir, device, max_samples=3):
    ensure_dir(output_dir)
    model.eval()
    target_layer = model.encoder.layer4[-1]
    activations, gradients = {}, {}

    def fwd_hook(_, __, out):
        activations["value"] = out.detach()

    def bwd_hook(_, __, grad_out):
        gradients["value"] = grad_out[0].detach()

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)
    inv_mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    inv_std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    count = min(max_samples, len(dataset))
    for idx in range(count):
        x, y, sid = dataset[idx]
        if getattr(model, "fusion", "mean") == "concat":
            model.zero_grad(set_to_none=True)
            batch = x.unsqueeze(0).to(device)
            logits = model(batch)
            cls = int(logits.argmax(1).item())
            logits[0, cls].backward()
            act = activations["value"].detach()
            grad = gradients["value"].detach()
            weights = grad.mean(dim=(2, 3), keepdim=True)
            cam_all = (weights * act).sum(dim=1).relu()
            cam_all = torch.nn.functional.interpolate(
                cam_all.unsqueeze(1), size=x.shape[-2:], mode="bilinear", align_corners=False
            )[:, 0].cpu().numpy()
            fig, axes = plt.subplots(2, len(dataset.views), figsize=(3 * len(dataset.views), 6))
            for view_idx, view in enumerate(dataset.views):
                cam = cam_all[view_idx]
                cam = (cam - cam.min()) / max(cam.max() - cam.min(), 1e-12)
                img = (x[view_idx].cpu() * inv_std + inv_mean).clamp(0, 1).permute(1, 2, 0).numpy()
                axes[0, view_idx].imshow(img)
                axes[0, view_idx].set_title(view)
                axes[0, view_idx].axis("off")
                axes[1, view_idx].imshow(img)
                axes[1, view_idx].imshow(cam, cmap="jet", alpha=0.45)
                axes[1, view_idx].axis("off")
            fig.suptitle(f"{sid} true={int(y)} pred={cls}")
            fig.tight_layout()
            fig.savefig(Path(output_dir) / f"gradcam_{sid}.png", dpi=180)
            plt.close(fig)
            continue
        fig, axes = plt.subplots(2, len(dataset.views), figsize=(3 * len(dataset.views), 6))
        for view_idx, view in enumerate(dataset.views):
            model.zero_grad(set_to_none=True)
            one = x[view_idx:view_idx + 1].unsqueeze(0).to(device)
            logits = model(one)
            cls = int(logits.argmax(1).item())
            logits[0, cls].backward()
            act = activations["value"]
            grad = gradients["value"]
            weights = grad.mean(dim=(2, 3), keepdim=True)
            cam = (weights * act).sum(dim=1).relu()
            cam = torch.nn.functional.interpolate(cam.unsqueeze(1), size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
            cam = cam.cpu().numpy()
            cam = (cam - cam.min()) / max(cam.max() - cam.min(), 1e-12)
            img = (x[view_idx].cpu() * inv_std + inv_mean).clamp(0, 1).permute(1, 2, 0).numpy()
            axes[0, view_idx].imshow(img)
            axes[0, view_idx].set_title(view)
            axes[0, view_idx].axis("off")
            axes[1, view_idx].imshow(img)
            axes[1, view_idx].imshow(cam, cmap="jet", alpha=0.45)
            axes[1, view_idx].axis("off")
        fig.suptitle(f"{sid} true={int(y)}")
        fig.tight_layout()
        fig.savefig(Path(output_dir) / f"gradcam_{sid}.png", dpi=180)
        plt.close(fig)
    h1.remove()
    h2.remove()


def main():
    parser = argparse.ArgumentParser(description="Fine-tune multi-view classifier and export retrieval/Grad-CAM outputs.")
    parser.add_argument("--manifest_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--encoder_ckpt", default="")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--no_imagenet_init", action="store_true")
    parser.add_argument("--fusion", choices=["mean", "concat"], default="mean")
    parser.add_argument("--views", default=",".join(DEFAULT_VIEWS), help="Comma-separated manifest view prefixes.")
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--shuffle_train_labels", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    out = Path(args.output_dir)
    ensure_dir(out / "checkpoints")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = device.type == "cuda"
    views = [x.strip() for x in args.views.split(",") if x.strip()]
    train_ds = MultiViewDataset(
        args.manifest_csv,
        "train",
        args.image_size,
        views,
        augment=args.augment,
        shuffle_labels=args.shuffle_train_labels,
        seed=args.seed,
    )
    val_ds = MultiViewDataset(args.manifest_csv, "val", args.image_size, views)
    test_ds = MultiViewDataset(args.manifest_csv, "test", args.image_size, views)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=(device.type == "cuda"), persistent_workers=(args.num_workers > 0))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=(device.type == "cuda"))
    model = MultiViewClassifier(args.encoder_ckpt, pretrained=not args.no_imagenet_init,
                                freeze_encoder=args.freeze_encoder, fusion=args.fusion,
                                num_views=len(views), dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    best_loss, bad, best_epoch = float("inf"), 0, 0
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
            best_loss, bad, best_epoch = val["loss"], 0, epoch
            torch.save({"model": model.state_dict(), "args": vars(args)}, out / "checkpoints" / "best_model.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print(">>> Early stopping triggered.", flush=True)
                break

    model.load_state_dict(torch.load(out / "checkpoints" / "best_model.pt", map_location=device)["model"])
    test = eval_model(model, test_loader, device, criterion)
    cm = confusion_matrix(test["y_true"], test["y_pred"])
    retrieval = retrieval_metrics(test["features"], test["y_true"])
    metrics = {
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "test_loss": test["loss"],
        "test_acc": test["acc"],
        "test_precision_weighted": test["precision"],
        "test_recall_weighted": test["recall"],
        "test_f1_weighted": test["f1"],
        "confusion_matrix": cm.tolist(),
        "retrieval": retrieval,
        "classification_report": classification_report(test["y_true"], test["y_pred"], output_dict=True, zero_division=0),
        "args": vars(args),
        "total_time_min": (time.time() - started) / 60.0,
    }
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    save_json(metrics, out / "test_metrics.json")
    pred_df = pd.DataFrame({
        "sample_id": test["ids"],
        "y_true": test["y_true"],
        "y_pred": test["y_pred"],
        "prob_up": test["prob_up"],
    })
    pred_df["ticker"] = pred_df["sample_id"].astype(str).str.split("_").str[0]
    pred_df.to_csv(out / "test_predictions.csv", index=False, encoding="utf-8-sig")
    ticker_rows = []
    for ticker, g in pred_df.groupby("ticker"):
        ticker_rows.append({
            "ticker": ticker,
            "n": int(len(g)),
            "acc": float((g["y_true"].to_numpy() == g["y_pred"].to_numpy()).mean()),
            "positive_rate": float(g["y_true"].mean()),
            "pred_positive_rate": float(g["y_pred"].mean()),
        })
    pd.DataFrame(ticker_rows).sort_values("ticker").to_csv(out / "test_by_ticker.csv", index=False, encoding="utf-8-sig")
    with open(out / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(test["y_true"], test["y_pred"], zero_division=0))
    save_gradcam(model, test_ds, out / "gradcam", device=device, max_samples=4)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
