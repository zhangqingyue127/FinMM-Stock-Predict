# Uncertainty-aware Multi-view Expert Fusion for Candlestick Image Trend Classification

This folder contains the code, lightweight data, experiment results, figures, and paper source used for the image-only candlestick trend classification study.

## Scope

The package is intentionally limited to the paper pipeline:

- paper-style candlestick image reproduction;
- pure-image multi-view inputs: candlestick, moving average, RSI, and MACD;
- multi-view consistency pretraining and fusion baselines;
- uncertainty-aware mixture-of-experts fusion;
- overfitting and shuffled-label diagnostics;
- Grad-CAM and gate-weight visualizations;
- IEEE/Overleaf paper source.

It does not include earlier unrelated A-share crawling data, tabular baselines, or exploratory preprocessing artifacts.

## Folder Layout

- `code/`: Python scripts for dataset preparation, model training, multi-view fusion, U-MoE training, and figure generation.
- `data/yahoo_ohlc_10stocks/`: OHLC CSV files for ten U.S. stocks used to regenerate the image datasets.
- `results/`: selected JSON/CSV experiment outputs used in the paper.
- `figures/`: paper-ready figures, including multi-view examples, gate weights, and Grad-CAM cases.
- `paper/`: Markdown and LaTeX paper drafts, including the Overleaf IEEE source.
- `docs/`: experiment interpretation notes and overfitting diagnostics.

## Environment

Install the core dependencies:

```bash
pip install -r requirements.txt
```

Recommended runtime:

- Python 3.10+
- PyTorch with CUDA for model training
- torchvision
- pandas, numpy, matplotlib, scikit-learn, pillow

## Reproduction Outline

1. Generate paper-style 30-day candlestick images:

```bash
python code/13_prepare_pizon2025_image_dataset.py \
  --raw_dir data/yahoo_ohlc_10stocks \
  --out_root outputs/pizon2025_repro \
  --lookback 30
```

2. Train the single-view CNN/ResNet reproduction baseline:

```bash
python code/14_train_pizon2025_cnn.py \
  --manifest outputs/pizon2025_repro/manifest.csv \
  --out_dir outputs/reproduction_window_trend
```

3. Generate multi-view image data:

```bash
python code/15_prepare_pizon2025_multiview_dataset.py \
  --raw_dir data/yahoo_ohlc_10stocks \
  --out_root outputs/pizon2025_multiview \
  --lookback 30
```

4. Train multi-view baselines or U-MoE:

```bash
python code/22_train_uncertainty_moe_multiview.py \
  --manifest outputs/pizon2025_multiview/manifest.csv \
  --out_dir outputs/umoe_window_trend \
  --label_col y_window_trend
```

Exact command-line options may differ depending on the final dataset layout; run each script with `--help` for the supported arguments.

## Key Reported Results

Representative outputs are stored under `results/`.

- `results/reproduction/`: paper-style visible trend recognition vs. next-day prediction reproduction.
- `results/multiview_future1/`: pure-image multi-view future-label experiments.
- `results/multiview_windowtrend_ablation/`: mean fusion, concat fusion, composite view, and related ablations.
- `results/overfit_checks/`: regularization, frozen probe, and shuffled-label diagnostics.
- `results/umoe_runs/`: U-MoE variants with gate-weight summaries and Grad-CAM examples.
- `results/umoe_metrics_summary.csv`: compact comparison table for U-MoE variants.

## Paper Source

The Overleaf-ready IEEE source is:

```text
paper/main_overleaf_ieee_with_figures_refs_v4.tex
```

When uploading to Overleaf, upload this folder together with `figures/`, then set the main document to the `.tex` file above.

## Notes

The main scientific message is interpretability-oriented rather than claiming strong next-day predictability. The experiments show that image models can recognize visible chart trends, while strict next-day direction prediction remains close to random. U-MoE provides expert weights, expert entropies, and Grad-CAM evidence to explain both confident visible-trend decisions and uncertain future-prediction failures.
