import argparse
import glob
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd


TICKERS = ["AAPL", "TSLA", "MSFT", "AMZN", "NVDA", "META", "GOOG", "JPM", "AMD", "BAC"]
VIEWS = ["candle", "ma", "rsi", "macd"]


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_csv_dir(csv_dir):
    out = {}
    csv_dir = Path(csv_dir)
    for ticker in TICKERS:
        df = pd.read_csv(csv_dir / f"{ticker}.csv")
        df["Date"] = pd.to_datetime(df["Date"])
        out[ticker] = df[["Date", "Open", "High", "Low", "Close"]].dropna().sort_values("Date").reset_index(drop=True)
    return out


def temporal_split(date, train_end, val_end):
    if date <= train_end:
        return "train"
    if date <= val_end:
        return "val"
    return "test"


def main():
    parser = argparse.ArgumentParser(description="Reuse multi-view images and rebuild labels for another trend target.")
    parser.add_argument("--csv_dir", required=True)
    parser.add_argument("--image_root", required=True, help="Existing multiview dataset root containing views/<view>/...")
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--max_samples", type=int, default=5283)
    parser.add_argument("--label_mode", choices=["future_k", "window_trend"], default="window_trend")
    parser.add_argument("--future_k", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_root = Path(args.out_root)
    ensure_dir(out_root)
    data = load_csv_dir(args.csv_dir)
    all_dates = sorted(set(pd.concat([df["Date"] for df in data.values()]).dropna()))
    train_end = pd.Timestamp(all_dates[int(len(all_dates) * 0.70)])
    val_end = pd.Timestamp(all_dates[int(len(all_dates) * 0.85)])

    candidates = []
    for ticker, df in data.items():
        for start_idx in range(0, len(df) - args.window - args.future_k + 1):
            end_idx = start_idx + args.window - 1
            target_idx = end_idx + args.future_k
            candidates.append((ticker, start_idx, end_idx, target_idx, pd.Timestamp(df.iloc[end_idx]["Date"])))

    if args.max_samples > 0 and len(candidates) > args.max_samples:
        by_ticker = {t: [] for t in TICKERS}
        for item in candidates:
            by_ticker[item[0]].append(item)
        sampled = []
        per_ticker = args.max_samples // len(TICKERS)
        remainder = args.max_samples % len(TICKERS)
        for i, ticker in enumerate(TICKERS):
            n = min(len(by_ticker[ticker]), per_ticker + (1 if i < remainder else 0))
            sampled.extend(rng.sample(by_ticker[ticker], n))
        candidates = sorted(sampled, key=lambda x: (x[4], x[0], x[1]))

    rows = []
    for n, (ticker, start_idx, end_idx, target_idx, end_date) in enumerate(candidates):
        df = data[ticker]
        start_close = float(df.iloc[start_idx]["Close"])
        current_close = float(df.iloc[end_idx]["Close"])
        future_close = float(df.iloc[target_idx]["Close"])
        label = int(current_close > start_close) if args.label_mode == "window_trend" else int(future_close > current_close)
        split = temporal_split(end_date, train_end, val_end)
        sample_id = f"{ticker}_{end_date.strftime('%Y%m%d')}_{n:06d}"
        row = {
            "sample_id": sample_id,
            "ticker": ticker,
            "start_date": pd.Timestamp(df.iloc[start_idx]["Date"]).strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "target_date": pd.Timestamp(df.iloc[target_idx]["Date"]).strftime("%Y-%m-%d"),
            "label": label,
            "class_name": "uptrend" if label == 1 else "downtrend",
            "split": split,
        }
        for view in VIEWS:
            matches = glob.glob(str(Path(args.image_root) / "views" / view / "*" / "*" / f"{sample_id}.png"))
            if len(matches) != 1:
                raise FileNotFoundError(f"Expected one image for {view}/{sample_id}, got {len(matches)}")
            row[f"{view}_path"] = matches[0]
        rows.append(row)

    manifest = pd.DataFrame(rows)
    manifest.to_csv(out_root / "manifest.csv", index=False, encoding="utf-8-sig")
    summary = {
        "image_root": args.image_root,
        "label_mode": args.label_mode,
        "future_k": args.future_k,
        "num_samples": int(len(manifest)),
        "split_counts": {k: int(v) for k, v in manifest["split"].value_counts().to_dict().items()},
        "class_counts": {k: int(v) for k, v in manifest["class_name"].value_counts().to_dict().items()},
        "manifest": str(out_root / "manifest.csv"),
    }
    with open(out_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
