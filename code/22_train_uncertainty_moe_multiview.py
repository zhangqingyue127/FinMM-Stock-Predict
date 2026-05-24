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
import torch.nn.functional as F
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
    def __init__(self, manifest_csv, split, image_size, views, augment=False):
        df = pd.read_csv(manifest_csv)
        self.df = df[df["split"] == split].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"No rows for split={split}")
        self.views = views
        ops = [transforms.Resize((image_size, image_size))]
        if augment:
            ops.extend([
                transforms.RandomAffine(degrees=0, translate=(0.025, 0.025), scale=(0.98, 1.02)),
                transforms.RandomApply([transforms.ColorJitter(0.08, 0.08, 0.02, 0.0)], p=0.35),
            ])
        ops.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        self.transform = transforms.Compose(ops)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        xs = [self.transform(Image.open(row[f"{v}_path"]).convert("RGB")) for v in self.views]
        return torch.stack(xs, dim=0), int(row["label"]), row["sample_id"]


def load_encoder_checkpoint(encoder, checkpoint_path):
    if not checkpoint_path:
        return
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("encoder_state_dict", ckpt.get("model_state_dict", ckpt))
    cleaned = {}
    for key, value in state.items():
        k = key.replace("module.", "")
        if k.startswith("encoder."):
            k = k[len("encoder."):]
        if k.startswith("fc."):
            continue
        cleaned[k] = value
    missing, unexpected = encoder.load_state_dict(cleaned, strict=False)
    print(f">>> loaded encoder ckpt={checkpoint_path} tensors={len(cleaned)} missing={len(missing)} unexpected={len(unexpected)}", flush=True)


class UncertaintyMoE(nn.Module):
    def __init__(
        self,
        num_views,
        num_classes=2,
        pretrained=True,
        encoder_ckpt="",
        dropout=0.35,
        uncertainty_lambda=1.0,
        uncertainty_aware=True,
        separate_experts=False,
        freeze_encoder=False,
        global_gate=False,
        residual_alpha=0.0,
    ):
        super().__init__()
        self.num_views = num_views
        self.num_classes = num_classes
        self.uncertainty_lambda = uncertainty_lambda
        self.uncertainty_aware = uncertainty_aware
        self.separate_experts = separate_experts
        self.global_gate = global_gate
        self.residual_alpha = residual_alpha

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.resnet18(weights=weights)
        dim = base.fc.in_features
        base.fc = nn.Identity()
        load_encoder_checkpoint(base, encoder_ckpt)

        if separate_experts:
            self.encoders = nn.ModuleList()
            self.expert_heads = nn.ModuleList()
            for _ in range(num_views):
                enc = models.resnet18(weights=None)
                enc.fc = nn.Identity()
                enc.load_state_dict(base.state_dict(), strict=True)
                self.encoders.append(enc)
                self.expert_heads.append(nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, num_classes)))
        else:
            self.encoder = base
            self.expert_heads = nn.ModuleList([
                nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, num_classes))
                for _ in range(num_views)
            ])

        if global_gate:
            self.gate = nn.Sequential(
                nn.Linear(dim * num_views, dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(dim, num_views),
            )
        else:
            self.gate = nn.Sequential(
                nn.Linear(dim, dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(dim // 2, 1),
            )

        self.concat_head = nn.Sequential(
            nn.Linear(dim * num_views, dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes),
        )

        if freeze_encoder:
            encoders = self.encoders if separate_experts else [self.encoder]
            for enc in encoders:
                for p in enc.parameters():
                    p.requires_grad = False

    def encode_views(self, x):
        b, v, c, h, w = x.shape
        if self.separate_experts:
            feats = []
            for i, enc in enumerate(self.encoders):
                feats.append(enc(x[:, i]))
            return torch.stack(feats, dim=1)
        feat = self.encoder(x.reshape(b * v, c, h, w)).reshape(b, v, -1)
        return feat

    def forward(self, x):
        feats = self.encode_views(x)
        expert_logits = torch.stack(
            [self.expert_heads[i](feats[:, i]) for i in range(self.num_views)],
            dim=1,
        )
        expert_prob = torch.softmax(expert_logits, dim=-1)
        entropy = -(expert_prob * torch.log(expert_prob.clamp_min(1e-8))).sum(dim=-1)
        if self.global_gate:
            gate_score = self.gate(feats.reshape(feats.size(0), -1))
        else:
            gate_score = self.gate(feats).squeeze(-1)
        if self.uncertainty_aware:
            gate_score = gate_score - self.uncertainty_lambda * entropy
        weights = torch.softmax(gate_score, dim=1)
        moe_prob = (weights.unsqueeze(-1) * expert_prob).sum(dim=1)
        concat_logits = self.concat_head(feats.reshape(feats.size(0), -1))
        concat_prob = torch.softmax(concat_logits, dim=-1)
        if self.residual_alpha > 0:
            prob = (1.0 - self.residual_alpha) * moe_prob + self.residual_alpha * concat_prob
        else:
            prob = moe_prob
        logits = torch.log(prob.clamp_min(1e-8))
        return {
            "logits": logits,
            "prob": prob,
            "moe_prob": moe_prob,
            "concat_logits": concat_logits,
            "concat_prob": concat_prob,
            "weights": weights,
            "entropy": entropy,
            "expert_logits": expert_logits,
            "expert_prob": expert_prob,
            "features": feats,
        }


def compute_losses(outputs, y, beta_aux, gamma_div, eta_cons, delta_concat):
    loss_cls = F.nll_loss(outputs["logits"], y)
    expert_logits = outputs["expert_logits"]
    b, v, c = expert_logits.shape
    loss_aux = F.cross_entropy(expert_logits.reshape(b * v, c), y[:, None].repeat(1, v).reshape(-1))
    loss_concat = F.cross_entropy(outputs["concat_logits"], y)

    feats = F.normalize(outputs["features"], dim=-1)
    sim = torch.matmul(feats, feats.transpose(1, 2))
    eye = torch.eye(v, device=feats.device, dtype=torch.bool).unsqueeze(0)
    loss_div = sim.masked_select(~eye).mean()

    p = outputs["prob"].detach().clamp_min(1e-8)
    expert_logp = torch.log(outputs["expert_prob"].clamp_min(1e-8))
    p_expand = p[:, None, :].expand_as(outputs["expert_prob"])
    loss_cons = F.kl_div(expert_logp, p_expand, reduction="batchmean") / v

    total = loss_cls + beta_aux * loss_aux + gamma_div * loss_div + eta_cons * loss_cons + delta_concat * loss_concat
    return total, {
        "loss_cls": float(loss_cls.detach().cpu()),
        "loss_aux": float(loss_aux.detach().cpu()),
        "loss_div": float(loss_div.detach().cpu()),
        "loss_cons": float(loss_cons.detach().cpu()),
        "loss_concat": float(loss_concat.detach().cpu()),
    }


def eval_model(model, loader, device, criterion):
    model.eval()
    y_true, y_pred, ids = [], [], []
    probs, weights, entropies, feats_all = [], [], [], []
    total_loss, total = 0.0, 0
    with torch.no_grad():
        for x, y, sid in loader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            loss = criterion(out["logits"], y)
            pred = out["prob"].argmax(dim=1)
            total_loss += loss.item() * x.size(0)
            total += x.size(0)
            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())
            ids.extend(list(sid))
            probs.extend(out["prob"][:, 1].cpu().numpy().tolist())
            weights.append(out["weights"].cpu().numpy())
            entropies.append(out["entropy"].cpu().numpy())
            pooled_feat = (out["weights"].unsqueeze(-1) * out["features"]).sum(dim=1)
            feats_all.append(pooled_feat.cpu().numpy())
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
        "prob_up": np.array(probs),
        "ids": ids,
        "weights": np.concatenate(weights, axis=0),
        "entropies": np.concatenate(entropies, axis=0),
        "features": np.concatenate(feats_all, axis=0),
    }


def retrieval_metrics(features, labels, ks=(1, 5, 10)):
    x = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    sim = x @ x.T
    np.fill_diagonal(sim, -np.inf)
    order = np.argsort(-sim, axis=1)
    out = {}
    for k in ks:
        out[f"retrieval_hit_at_{k}"] = float((labels[order[:, :k]] == labels[:, None]).any(axis=1).mean())
    return out


def save_gradcam(model, dataset, output_dir, device, views, max_samples=4):
    ensure_dir(output_dir)
    model.eval()
    target_encoder = model.encoders[0] if getattr(model, "separate_experts", False) else model.encoder
    target_layer = target_encoder.layer4[-1]
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
        batch = x.unsqueeze(0).to(device)
        out = model(batch)
        cls = int(out["prob"].argmax(dim=1).item())
        out["logits"][0, cls].backward()
        act = activations["value"].detach()
        grad = gradients["value"].detach()
        view_count = len(views)
        if act.size(0) != view_count:
            # Separate-expert Grad-CAM hooks only the first view. Keep the figure useful rather than failing.
            act = act.repeat(view_count, 1, 1, 1)
            grad = grad.repeat(view_count, 1, 1, 1)
        cam_all = (grad.mean(dim=(2, 3), keepdim=True) * act).sum(dim=1).relu()
        cam_all = F.interpolate(cam_all.unsqueeze(1), size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        cam_all = cam_all.cpu().numpy()
        fig, axes = plt.subplots(2, view_count, figsize=(3 * view_count, 6))
        gate = out["weights"][0].detach().cpu().numpy()
        unc = out["entropy"][0].detach().cpu().numpy()
        for i, view in enumerate(views):
            cam = cam_all[i]
            cam = (cam - cam.min()) / max(cam.max() - cam.min(), 1e-12)
            img = (x[i].cpu() * inv_std + inv_mean).clamp(0, 1).permute(1, 2, 0).numpy()
            axes[0, i].imshow(img)
            axes[0, i].set_title(f"{view}\nw={gate[i]:.2f} H={unc[i]:.2f}")
            axes[0, i].axis("off")
            axes[1, i].imshow(img)
            axes[1, i].imshow(cam, cmap="jet", alpha=0.45)
            axes[1, i].axis("off")
        fig.suptitle(f"{sid} true={int(y)} pred={cls}")
        fig.tight_layout()
        fig.savefig(Path(output_dir) / f"umoe_gradcam_{sid}.png", dpi=180)
        plt.close(fig)
    h1.remove()
    h2.remove()


def plot_gate_summary(weights, views, output_dir):
    ensure_dir(output_dir)
    means = weights.mean(axis=0)
    stds = weights.std(axis=0)
    df = pd.DataFrame({"view": views, "mean_weight": means, "std_weight": stds})
    df.to_csv(Path(output_dir) / "gate_weight_summary.csv", index=False)
    plt.figure(figsize=(7, 4))
    plt.bar(views, means, yerr=stds, capsize=4, color=["#2563eb", "#16a34a", "#f97316", "#7c3aed"][:len(views)])
    plt.ylabel("Gate weight")
    plt.ylim(0, max(0.5, float((means + stds).max()) * 1.25))
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "gate_weight_summary.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Uncertainty-aware multi-view mixture-of-experts for candlestick images.")
    parser.add_argument("--manifest_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--encoder_ckpt", default="")
    parser.add_argument("--views", default=",".join(DEFAULT_VIEWS))
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--uncertainty_lambda", type=float, default=0.5)
    parser.add_argument("--beta_aux", type=float, default=0.3)
    parser.add_argument("--gamma_div", type=float, default=0.02)
    parser.add_argument("--eta_cons", type=float, default=0.05)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--no_imagenet_init", action="store_true")
    parser.add_argument("--no_uncertainty", action="store_true")
    parser.add_argument("--separate_experts", action="store_true")
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--global_gate", action="store_true")
    parser.add_argument("--residual_alpha", type=float, default=0.0,
                        help="Blend final probability with a concat residual branch: p=(1-a)*p_moe+a*p_concat.")
    parser.add_argument("--delta_concat", type=float, default=0.0,
                        help="Auxiliary CE weight for the concat residual branch.")
    args = parser.parse_args()

    set_seed(args.seed)
    out = Path(args.output_dir)
    ensure_dir(out / "checkpoints")
    views = [x.strip() for x in args.views.split(",") if x.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = device.type == "cuda"

    train_ds = MultiViewDataset(args.manifest_csv, "train", args.image_size, views, augment=args.augment)
    val_ds = MultiViewDataset(args.manifest_csv, "val", args.image_size, views)
    test_ds = MultiViewDataset(args.manifest_csv, "test", args.image_size, views)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=(device.type == "cuda"), persistent_workers=(args.num_workers > 0))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=(device.type == "cuda"))

    model = UncertaintyMoE(
        num_views=len(views),
        pretrained=not args.no_imagenet_init,
        encoder_ckpt=args.encoder_ckpt,
        dropout=args.dropout,
        uncertainty_lambda=args.uncertainty_lambda,
        uncertainty_aware=not args.no_uncertainty,
        separate_experts=args.separate_experts,
        freeze_encoder=args.freeze_encoder,
        global_gate=args.global_gate,
        residual_alpha=args.residual_alpha,
    ).to(device)
    criterion = nn.NLLLoss()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    best_val_loss, best_epoch, bad = float("inf"), 0, 0
    history = []
    started = time.time()
    print(f">>> device={device} train/val/test={len(train_ds)}/{len(val_ds)}/{len(test_ds)} views={views}", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total, correct = 0.0, 0, 0
        aux_sums = {"loss_cls": 0.0, "loss_aux": 0.0, "loss_div": 0.0, "loss_cons": 0.0, "loss_concat": 0.0}
        for x, y, _ in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            outputs = model(x)
            if args.label_smoothing > 0:
                # Replace CE with smoothed CE on log probabilities for the main path.
                loss_cls = F.cross_entropy(outputs["logits"], y, label_smoothing=args.label_smoothing)
                loss, parts = compute_losses(outputs, y, args.beta_aux, args.gamma_div, args.eta_cons, args.delta_concat)
                loss = loss - F.nll_loss(outputs["logits"], y) + loss_cls
                parts["loss_cls"] = float(loss_cls.detach().cpu())
            else:
                loss, parts = compute_losses(outputs, y, args.beta_aux, args.gamma_div, args.eta_cons, args.delta_concat)
            loss.backward()
            opt.step()
            pred = outputs["prob"].argmax(dim=1)
            total_loss += loss.item() * x.size(0)
            total += x.size(0)
            correct += (pred == y).sum().item()
            for k in aux_sums:
                aux_sums[k] += parts[k] * x.size(0)

        val = eval_model(model, val_loader, device, criterion)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(total, 1),
            "train_acc": correct / max(total, 1),
            "val_loss": val["loss"],
            "val_acc": val["acc"],
            "val_f1": val["f1"],
        }
        for k, v in aux_sums.items():
            row[f"train_{k}"] = v / max(total, 1)
        history.append(row)
        print(
            f"[Epoch {epoch:03d}/{args.epochs:03d}] "
            f"train_loss={row['train_loss']:.4f} train_acc={row['train_acc']:.4f} "
            f"val_loss={row['val_loss']:.4f} val_acc={row['val_acc']:.4f} val_f1={row['val_f1']:.4f} "
            f"div={row['train_loss_div']:.4f}",
            flush=True,
        )
        if val["loss"] < best_val_loss:
            best_val_loss, best_epoch, bad = val["loss"], epoch, 0
            torch.save({"model": model.state_dict(), "args": vars(args), "views": views}, out / "checkpoints" / "best_model.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print(">>> Early stopping triggered.", flush=True)
                break

    ckpt = torch.load(out / "checkpoints" / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    test = eval_model(model, test_loader, device, criterion)
    retrieval = retrieval_metrics(test["features"], test["y_true"])
    metrics = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_loss": test["loss"],
        "test_acc": test["acc"],
        "test_precision_weighted": test["precision"],
        "test_recall_weighted": test["recall"],
        "test_f1_weighted": test["f1"],
        "confusion_matrix": confusion_matrix(test["y_true"], test["y_pred"]).tolist(),
        "retrieval": retrieval,
        "classification_report": classification_report(test["y_true"], test["y_pred"], output_dict=True, zero_division=0),
        "mean_gate_weights": {view: float(w) for view, w in zip(views, test["weights"].mean(axis=0))},
        "mean_expert_entropy": {view: float(e) for view, e in zip(views, test["entropies"].mean(axis=0))},
        "args": vars(args),
        "total_time_min": (time.time() - started) / 60.0,
    }
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    pred_df = pd.DataFrame({
        "sample_id": test["ids"],
        "y_true": test["y_true"],
        "y_pred": test["y_pred"],
        "prob_up": test["prob_up"],
    })
    pred_df["ticker"] = pred_df["sample_id"].astype(str).str.split("_").str[0]
    for i, view in enumerate(views):
        pred_df[f"gate_{view}"] = test["weights"][:, i]
        pred_df[f"entropy_{view}"] = test["entropies"][:, i]
    pred_df.to_csv(out / "test_predictions_with_gates.csv", index=False, encoding="utf-8-sig")
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
    save_json(metrics, out / "test_metrics.json")
    plot_gate_summary(test["weights"], views, out)
    save_gradcam(model, test_ds, out / "gradcam", device, views, max_samples=4)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
