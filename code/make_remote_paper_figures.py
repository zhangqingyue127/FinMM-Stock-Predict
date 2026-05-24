import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


manifest = "/root/autodl-tmp/pizon2025_multiview_windowtrend/manifest.csv"
df = pd.read_csv(manifest)
rows = []
for label in [1, 0]:
    rows.extend(df[(df["split"] == "test") & (df["label"] == label)].head(2).to_dict("records"))

views = ["candle", "ma", "rsi", "macd"]
fig, axes = plt.subplots(len(rows), len(views), figsize=(9.5, 6.2))
for r, row in enumerate(rows):
    for c, view in enumerate(views):
        ax = axes[r, c]
        img = Image.open(row[f"{view}_path"]).convert("RGB")
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        if r == 0:
            ax.set_title(view.upper(), fontsize=10)
        if c == 0:
            ax.set_ylabel(f"{row['ticker']} {row['end_date']}\ny={row['label']}", fontsize=8)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)

fig.suptitle("Real multi-view candlestick image samples from the AutoDL dataset", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("/root/autodl-tmp/fig_candlestick_multiview_samples.png", dpi=220)
