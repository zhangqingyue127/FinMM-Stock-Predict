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
    hist = line - sig
    return line, sig, hist


def add_indicators(df):
    df = df.copy()
    df["RSI"] = rsi(df["Close"])
    df["MACD"], df["MACD_SIGNAL"], df["MACD_HIST"] = macd(df["Close"])
    ma = df["Close"].rolling(20, min_periods=5).mean()
    std = df["Close"].rolling(20, min_periods=5).std()
    df["CHANNEL_MID"] = ma
    df["CHANNEL_UP"] = ma + 2 * std
    df["CHANNEL_LOW"] = ma - 2 * std
    return df


def draw_candles(ax, data):
    x = np.arange(len(data))
    for i, (_, row) in enumerate(data.iterrows()):
        o, h, l, c = float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"])
        color = "#16a34a" if c >= o else "#dc2626"
        ax.vlines(i, l, h, color=color, linewidth=0.8)
        bottom = min(o, c)
        height = max(abs(c - o), max(h - l, 1e-6) * 0.015)
        ax.add_patch(plt.Rectangle((i - 0.32, bottom), 0.64, height, facecolor=color, edgecolor=color, linewidth=0.5))
    ax.plot(x, data["CHANNEL_MID"], color="#2563eb", linewidth=0.65, alpha=0.85)
    ax.fill_between(
        x,
        data["CHANNEL_LOW"].to_numpy(dtype=float),
        data["CHANNEL_UP"].to_numpy(dtype=float),
        color="#60a5fa",
        alpha=0.16,
        linewidth=0,
    )
    ax.set_xlim(-0.7, len(data) - 0.3)
    low = float(data["Low"].min())
    high = float(data["High"].max())
    pad = max((high - low) * 0.08, 1e-4)
    ax.set_ylim(low - pad, high + pad)
    ax.axis("off")


def render_chart(data, out_path, image_size=100):
    fig = plt.figure(figsize=(image_size / 100, image_size / 100), dpi=100, facecolor="white")
    ax_price = fig.add_axes([0.02, 0.37, 0.96, 0.61])
    ax_rsi = fig.add_axes([0.02, 0.20, 0.96, 0.14])
    ax_macd = fig.add_axes([0.02, 0.02, 0.96, 0.15])

    draw_candles(ax_price, data)

    x = np.arange(len(data))
    ax_rsi.plot(x, data["RSI"].fillna(50), color="#2563eb", linewidth=0.7)
    ax_rsi.axhline(70, color="#9ca3af", linewidth=0.35)
    ax_rsi.axhline(30, color="#9ca3af", linewidth=0.35)
    ax_rsi.set_xlim(-0.7, len(data) - 0.3)
    ax_rsi.set_ylim(0, 100)
    ax_rsi.axis("off")

    hist = data["MACD_HIST"].fillna(0).to_numpy(dtype=float)
    colors = np.where(hist >= 0, "#16a34a", "#dc2626")
    ax_macd.bar(x, hist, color=colors, width=0.65, linewidth=0)
    ax_macd.plot(x, data["MACD"].fillna(0), color="#111827", linewidth=0.55)
    ax_macd.plot(x, data["MACD_SIGNAL"].fillna(0), color="#f97316", linewidth=0.55)
    ax_macd.set_xlim(-0.7, len(data) - 0.3)
    m = max(abs(float(np.nanmin(hist))), abs(float(np.nanmax(hist))), 1e-6)
    ax_macd.set_ylim(-m * 1.25, m * 1.25)
    ax_macd.axis("off")

    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def load_csv_dir(csv_dir, tickers):
    out = {}
    csv_dir = Path(csv_dir)
    for ticker in tickers:
        path = csv_dir / f"{ticker}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        df["Date"] = pd.to_datetime(df["Date"])
        needed = ["Date", "Open", "High", "Low", "Close"]
        df = df[needed].dropna().sort_values("Date").reset_index(drop=True)
        out[ticker] = add_indicators(df)
    return out


def download_yahoo(tickers, start, end):
    import yfinance as yf

    out = {}
    for ticker in tickers:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"])
        needed = ["Date", "Open", "High", "Low", "Close"]
        df = df[needed].dropna().sort_values("Date").reset_index(drop=True)
        out[ticker] = add_indicators(df)
    return out


def temporal_split(date, train_end, val_end):
    if date <= train_end:
        return "train"
    if date <= val_end:
        return "val"
    return "test"


def main():
    parser = argparse.ArgumentParser(description="Reproduce Pizon et al. 2025 candlestick-image trend dataset.")
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--start", default="2020-02-20")
    parser.add_argument("--end", default="2023-12-19", help="Exclusive end date; paper reports data through 2023-12-18.")
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--image_size", type=int, default=100)
    parser.add_argument("--max_samples", type=int, default=5283)
    parser.add_argument("--csv_dir", default="", help="Optional directory containing AAPL.csv, TSLA.csv, ... downloaded from Yahoo.")
    parser.add_argument("--label_mode", choices=["next_day", "window_trend"], default="next_day")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_root = Path(args.out_root)
    image_root = out_root / "images"
    ensure_dir(image_root)

    data = load_csv_dir(args.csv_dir, TICKERS) if args.csv_dir else download_yahoo(TICKERS, args.start, args.end)
    all_dates = sorted(set(pd.concat([df["Date"] for df in data.values()]).dropna()))
    train_end = pd.Timestamp(all_dates[int(len(all_dates) * 0.70)])
    val_end = pd.Timestamp(all_dates[int(len(all_dates) * 0.85)])

    rows = []
    candidates = []
    for ticker, df in data.items():
        for start_idx in range(0, len(df) - args.window):
            end_idx = start_idx + args.window - 1
            target_idx = start_idx + args.window
            start_close = float(df.iloc[start_idx]["Close"])
            current_close = float(df.iloc[end_idx]["Close"])
            next_close = float(df.iloc[target_idx]["Close"])
            if args.label_mode == "window_trend":
                label = 1 if current_close > start_close else 0
            else:
                label = 1 if next_close > current_close else 0
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
            items = by_ticker[ticker]
            n = min(len(items), per_ticker + (1 if i < remainder else 0))
            sampled.extend(rng.sample(items, n))
        candidates = sorted(sampled, key=lambda x: (x[4], x[0], x[1]))

    for n, (ticker, start_idx, end_idx, target_idx, end_date, label) in enumerate(candidates):
        df = data[ticker]
        window_df = df.iloc[start_idx:end_idx + 1].copy()
        split = temporal_split(end_date, train_end, val_end)
        class_name = "uptrend" if label == 1 else "downtrend"
        sample_id = f"{ticker}_{end_date.strftime('%Y%m%d')}_{n:06d}"
        out_path = image_root / split / class_name / f"{sample_id}.png"
        ensure_dir(out_path.parent)
        render_chart(window_df, out_path, args.image_size)
        rows.append({
            "sample_id": sample_id,
            "ticker": ticker,
            "start_date": pd.Timestamp(window_df.iloc[0]["Date"]).strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "target_date": pd.Timestamp(df.iloc[target_idx]["Date"]).strftime("%Y-%m-%d"),
            "current_close": current_close,
            "next_close": next_close,
            "start_close": start_close,
            "label": label,
            "class_name": class_name,
            "split": split,
            "image_path": str(out_path),
        })
        if (n + 1) % 500 == 0 or (n + 1) == len(candidates):
            print(f"[render] {n+1}/{len(candidates)}", flush=True)

    manifest = pd.DataFrame(rows)
    manifest_path = out_root / "manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    summary = {
        "paper": "Image-based time series trend classification using deep learning: A candlestick chart approach",
        "tickers": TICKERS,
        "start": args.start,
        "end_exclusive": args.end,
        "window": args.window,
        "image_size": args.image_size,
        "max_samples": args.max_samples,
        "label_mode": args.label_mode,
        "num_samples": int(len(manifest)),
        "split_counts": {k: int(v) for k, v in manifest["split"].value_counts().to_dict().items()},
        "class_counts": {k: int(v) for k, v in manifest["class_name"].value_counts().to_dict().items()},
        "ticker_counts": {k: int(v) for k, v in manifest["ticker"].value_counts().to_dict().items()},
        "manifest": str(manifest_path),
        "image_root": str(image_root),
    }
    save_json(summary, out_root / "summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
