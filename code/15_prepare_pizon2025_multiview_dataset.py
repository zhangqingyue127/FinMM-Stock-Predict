import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TICKERS = ["AAPL", "TSLA", "MSFT", "AMZN", "NVDA", "META", "GOOG", "JPM", "AMD", "BAC"]
VIEWS = ["candle", "ma", "rsi", "macd"]


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def rsi(close, window=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close, fast=12, slow=26, signal=9):
    line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def add_indicators(df):
    df = df.copy()
    close = df["Close"]
    df["MA5"] = close.rolling(5, min_periods=1).mean()
    df["MA10"] = close.rolling(10, min_periods=1).mean()
    df["MA20"] = close.rolling(20, min_periods=1).mean()
    df["RSI"] = rsi(close).bfill().fillna(50)
    df["MACD"], df["MACD_SIGNAL"], df["MACD_HIST"] = macd(close)
    return df


def load_csv_dir(csv_dir):
    out = {}
    csv_dir = Path(csv_dir)
    for ticker in TICKERS:
        path = csv_dir / f"{ticker}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df[["Date", "Open", "High", "Low", "Close"]].dropna().sort_values("Date").reset_index(drop=True)
        out[ticker] = add_indicators(df)
    return out


def temporal_split(date, train_end, val_end):
    if date <= train_end:
        return "train"
    if date <= val_end:
        return "val"
    return "test"


def draw_candles(ax, data, add_ma=False):
    x = np.arange(len(data))
    for i, (_, row) in enumerate(data.iterrows()):
        o, h, l, c = float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"])
        color = "#16a34a" if c >= o else "#dc2626"
        ax.vlines(i, l, h, color=color, linewidth=0.9)
        bottom = min(o, c)
        height = max(abs(c - o), max(h - l, 1e-6) * 0.018)
        ax.add_patch(plt.Rectangle((i - 0.32, bottom), 0.64, height, facecolor=color, edgecolor=color, linewidth=0.55))
    if add_ma:
        ax.plot(x, data["MA5"], color="#2563eb", linewidth=0.7)
        ax.plot(x, data["MA10"], color="#f97316", linewidth=0.7)
        ax.plot(x, data["MA20"], color="#111827", linewidth=0.7)
    low, high = float(data["Low"].min()), float(data["High"].max())
    pad = max((high - low) * 0.08, 1e-4)
    ax.set_xlim(-0.7, len(data) - 0.3)
    ax.set_ylim(low - pad, high + pad)
    ax.axis("off")


def finish_single_axis(fig, ax, out_path):
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def render_view(data, out_path, view, image_size):
    fig = plt.figure(figsize=(image_size / 100, image_size / 100), dpi=100, facecolor="white")
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    x = np.arange(len(data))
    if view == "candle":
        draw_candles(ax, data, add_ma=False)
    elif view == "ma":
        draw_candles(ax, data, add_ma=True)
    elif view == "rsi":
        ax.plot(x, data["RSI"].fillna(50), color="#2563eb", linewidth=1.15)
        ax.axhline(70, color="#9ca3af", linestyle="--", linewidth=0.6)
        ax.axhline(30, color="#9ca3af", linestyle="--", linewidth=0.6)
        ax.fill_between(x, 30, 70, color="#bfdbfe", alpha=0.20, linewidth=0)
        ax.set_xlim(-0.7, len(data) - 0.3)
        ax.set_ylim(0, 100)
        ax.axis("off")
    elif view == "macd":
        hist = data["MACD_HIST"].fillna(0).to_numpy(dtype=float)
        colors = np.where(hist >= 0, "#16a34a", "#dc2626")
        ax.bar(x, hist, color=colors, width=0.65, linewidth=0)
        ax.plot(x, data["MACD"].fillna(0), color="#111827", linewidth=0.9)
        ax.plot(x, data["MACD_SIGNAL"].fillna(0), color="#f97316", linewidth=0.9)
        m = max(abs(float(np.nanmin(hist))), abs(float(np.nanmax(hist))), 1e-6)
        ax.set_xlim(-0.7, len(data) - 0.3)
        ax.set_ylim(-m * 1.35, m * 1.35)
        ax.axis("off")
    else:
        raise ValueError(f"unknown view: {view}")
    finish_single_axis(fig, ax, out_path)


def main():
    parser = argparse.ArgumentParser(description="Build pure-image multi-view candlestick dataset.")
    parser.add_argument("--csv_dir", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--start", default="2020-02-20")
    parser.add_argument("--end", default="2023-12-19")
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--max_samples", type=int, default=5283)
    parser.add_argument("--label_mode", choices=["future_k", "window_trend"], default="future_k")
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
            start_close = float(df.iloc[start_idx]["Close"])
            current_close = float(df.iloc[end_idx]["Close"])
            future_close = float(df.iloc[target_idx]["Close"])
            label = int(current_close > start_close) if args.label_mode == "window_trend" else int(future_close > current_close)
            end_date = pd.Timestamp(df.iloc[end_idx]["Date"])
            candidates.append((ticker, start_idx, end_idx, target_idx, end_date, label))

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
    for n, (ticker, start_idx, end_idx, target_idx, end_date, label) in enumerate(candidates):
        df = data[ticker]
        window_df = df.iloc[start_idx:end_idx + 1].copy()
        split = temporal_split(end_date, train_end, val_end)
        class_name = "uptrend" if label == 1 else "downtrend"
        sample_id = f"{ticker}_{end_date.strftime('%Y%m%d')}_{n:06d}"
        row = {
            "sample_id": sample_id,
            "ticker": ticker,
            "start_date": pd.Timestamp(window_df.iloc[0]["Date"]).strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "target_date": pd.Timestamp(df.iloc[target_idx]["Date"]).strftime("%Y-%m-%d"),
            "label": label,
            "class_name": class_name,
            "split": split,
        }
        for view in VIEWS:
            out_path = out_root / "views" / view / split / class_name / f"{sample_id}.png"
            ensure_dir(out_path.parent)
            render_view(window_df, out_path, view=view, image_size=args.image_size)
            row[f"{view}_path"] = str(out_path)
        rows.append(row)
        if (n + 1) % 500 == 0 or n + 1 == len(candidates):
            print(f"[render] {n + 1}/{len(candidates)}", flush=True)

    manifest = pd.DataFrame(rows)
    manifest.to_csv(out_root / "manifest.csv", index=False, encoding="utf-8-sig")
    summary = {
        "title": "Multi-view Candlestick Image Representation Learning for Interpretable Stock Trend Classification",
        "tickers": TICKERS,
        "views": VIEWS,
        "window": args.window,
        "image_size": args.image_size,
        "label_mode": args.label_mode,
        "future_k": args.future_k,
        "num_samples": int(len(manifest)),
        "split_counts": {k: int(v) for k, v in manifest["split"].value_counts().to_dict().items()},
        "class_counts": {k: int(v) for k, v in manifest["class_name"].value_counts().to_dict().items()},
        "manifest": str(out_root / "manifest.csv"),
    }
    save_json(summary, out_root / "summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
