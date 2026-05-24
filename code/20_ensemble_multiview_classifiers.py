import argparse
import json
import random
from pathlib import Path

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


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MultiViewDataset(Dataset):
    def __init__(self, manifest_csv, split, image_size):
        df = pd.read_csv(manifest_csv)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        xs = [self.transform(Image.open(row[f"{v}_path"]).convert("RGB")) for v in VIEWS]
        return torch.stack(xs, dim=0), int(row["label"]), row["sample_id"]


class MultiViewClassifier(nn.Module):
    def __init__(self, fusion="concat", pretrained=False, num_classes=2):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        encoder = models.resnet18(weights=weights)
        dim = encoder.fc.in_features
        encoder.fc = nn.Identity()
        self.encoder = encoder
        self.fusion = fusion
        if fusion == "concat":
            self.classifier = nn.Sequential(
                nn.Linear(dim * len(VIEWS), dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25),
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


def load_model(run_dir, device):
    run_dir = Path(run_dir)
    metrics_path = run_dir / "test_metrics.json"
    fusion = "mean"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        fusion = metrics.get("args", {}).get("fusion", "mean")
    model = MultiViewClassifier(fusion=fusion, pretrained=False).to(device)
    ckpt = torch.load(run_dir / "checkpoints" / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, fusion


def main():
    parser = argparse.ArgumentParser(description="Average probabilities from several trained multi-view classifiers.")
    parser.add_argument("--manifest_csv", required=True)
    parser.add_argument("--run_dirs", required=True, help="Comma-separated run directories.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = MultiViewDataset(args.manifest_csv, args.split, args.image_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"))
    run_dirs = [x.strip() for x in args.run_dirs.split(",") if x.strip()]
    models_and_fusions = [load_model(run_dir, device) for run_dir in run_dirs]
    print(">>> ensemble members:", [(str(run_dirs[i]), f) for i, (_, f) in enumerate(models_and_fusions)], flush=True)

    y_true, y_pred, ids = [], [], []
    probs_all = []
    with torch.no_grad():
        for x, y, sid in loader:
            x = x.to(device)
            probs = []
            for model, _ in models_and_fusions:
                probs.append(torch.softmax(model(x), dim=1))
            avg_prob = torch.stack(probs, dim=0).mean(dim=0)
            pred = avg_prob.argmax(1)
            y_true.extend(y.numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())
            probs_all.extend(avg_prob[:, 1].cpu().numpy().tolist())
            ids.extend(list(sid))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    metrics = {
        "split": args.split,
        "run_dirs": run_dirs,
        "test_acc": float((y_true == y_pred).mean()),
        "test_precision_weighted": float(precision),
        "test_recall_weighted": float(recall),
        "test_f1_weighted": float(f1),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }
    pd.DataFrame({"sample_id": ids, "y_true": y_true, "y_pred": y_pred, "prob_up": probs_all}).to_csv(
        out / f"{args.split}_predictions.csv", index=False
    )
    save_json(metrics, out / f"{args.split}_ensemble_metrics.json")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
