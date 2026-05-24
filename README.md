# Uncertainty-aware Multi-view Expert Fusion for Candlestick Image Trend Classification

This repository contains the code, lightweight data, selected results, visualizations, and paper source for an image-only financial chart study:

**Uncertainty-aware Multi-view Expert Fusion for Interpretable Candlestick Image Trend Classification**

The project focuses on a careful distinction that is easy to miss in chart-image stock papers:

- **visible window-trend recognition**: the model recognizes whether the input chart window itself trends up or down;
- **strict future prediction**: the model predicts the next trading day's direction from the image window.

Our experiments show that candlestick images can support very high visible-trend recognition accuracy, but strict next-day prediction remains close to random. The proposed U-MoE model is therefore positioned primarily as an **interpretable multi-view image fusion method**, not as a claim that static chart images reliably predict the next trading day.

## What Is Included

This repository is intentionally limited to the paper pipeline.

```text
code/        Dataset generation, reproduction baselines, multi-view models, U-MoE, Grad-CAM.
data/        Lightweight OHLC CSV files for ten U.S. stocks.
docs/        Experiment interpretation notes and overfitting diagnostics.
figures/     Paper-ready multi-view, gate-weight, and Grad-CAM figures.
paper/       Markdown and LaTeX paper drafts, including the Overleaf IEEE source.
results/     Selected JSON/CSV outputs used in the paper.
```

Earlier A-share crawling scripts, tabular baselines, raw exploratory EDA artifacts, and unrelated preprocessing experiments have been removed so that the repository matches the paper narrative.

## Core Idea

For each 30-trading-day stock window, the same market state is rendered as four pure image views:

1. raw candlestick chart;
2. candlestick chart with moving-average overlays;
3. RSI image;
4. MACD image.

U-MoE treats these views as image experts. Each expert produces a class probability, and an entropy-aware gate combines them:

```text
final prediction = weighted average of view-specific expert predictions
```

The model exposes three explanation levels:

- **expert probabilities**: what each view predicts;
- **expert entropies**: how uncertain each view is;
- **gate weights**: how much each view contributes to the fused decision.

Grad-CAM is used as a spatial explanation on top of the view-level explanations.

## Data

The folder `data/yahoo_ohlc_10stocks/` contains CSV files for:

```text
AAPL, AMD, AMZN, BAC, GOOG, JPM, META, MSFT, NVDA, TSLA
```

The paper setting uses daily OHLC data from 2020-02-20 to 2023-12-18 and constructs 30-day image windows. Generated image datasets are not committed because they can be regenerated from the CSV files.

## Environment

Install dependencies with:

```bash
pip install -r requirements.txt
```

Recommended runtime:

- Python 3.10+
- PyTorch with CUDA for training
- torchvision
- pandas, numpy, matplotlib, scikit-learn, pillow

## Quick Start

### 1. Reproduce the single-view candlestick baseline

Generate the visible window-trend dataset:

```bash
python code/13_prepare_pizon2025_image_dataset.py \
  --csv_dir data/yahoo_ohlc_10stocks \
  --out_root outputs/pizon2025_window_trend \
  --label_mode window_trend \
  --window 30 \
  --image_size 100
```

Train the lightweight CNN baseline:

```bash
python code/14_train_pizon2025_cnn.py \
  --data_root outputs/pizon2025_window_trend/images \
  --output_dir outputs/runs/window_trend_cnn \
  --epochs 50 \
  --batch_size 32
```

Generate the strict next-day dataset:

```bash
python code/13_prepare_pizon2025_image_dataset.py \
  --csv_dir data/yahoo_ohlc_10stocks \
  --out_root outputs/pizon2025_next_day \
  --label_mode next_day \
  --window 30 \
  --image_size 100
```

### 2. Generate the multi-view image dataset

For visible window-trend classification:

```bash
python code/15_prepare_pizon2025_multiview_dataset.py \
  --csv_dir data/yahoo_ohlc_10stocks \
  --out_root outputs/multiview_window_trend \
  --label_mode window_trend \
  --window 30 \
  --image_size 224
```

For strict next-day classification:

```bash
python code/15_prepare_pizon2025_multiview_dataset.py \
  --csv_dir data/yahoo_ohlc_10stocks \
  --out_root outputs/multiview_next_day \
  --label_mode future_k \
  --future_k 1 \
  --window 30 \
  --image_size 224
```

### 3. Train U-MoE

```bash
python code/22_train_uncertainty_moe_multiview.py \
  --manifest_csv outputs/multiview_window_trend/manifest.csv \
  --output_dir outputs/runs/uncertainty_moe \
  --epochs 30 \
  --batch_size 32 \
  --uncertainty_lambda 0.5
```

Useful variants:

```bash
# Vanilla MoE without entropy discounting
python code/22_train_uncertainty_moe_multiview.py \
  --manifest_csv outputs/multiview_window_trend/manifest.csv \
  --output_dir outputs/runs/vanilla_moe \
  --no_uncertainty

# Frozen encoder probe
python code/22_train_uncertainty_moe_multiview.py \
  --manifest_csv outputs/multiview_window_trend/manifest.csv \
  --output_dir outputs/runs/frozen_probe \
  --freeze_encoder

# Residual concat branch
python code/22_train_uncertainty_moe_multiview.py \
  --manifest_csv outputs/multiview_window_trend/manifest.csv \
  --output_dir outputs/runs/residual_umoe \
  --residual_alpha 0.5 \
  --delta_concat 0.1
```

Run each script with `--help` for the full list of arguments.

## Main Results

Selected result files are stored under `results/`.

### Task sanity check

| Task | Label | Model | Test Accuracy | Weighted F1 |
|---|---:|---|---:|---:|
| Visible trend recognition | window-trend | Lightweight CNN | 0.9837 | 0.9837 |
| Strict future prediction | next-day | Lightweight CNN | 0.4962 | 0.4757 |

This comparison is central: high chart-image accuracy can reflect recognition of a trend already visible in the input window, not reliable future prediction.

### U-MoE variants on window-trend classification

| Run | Accuracy | Weighted F1 | Retrieval Hit@1 | Candle | MA | RSI | MACD |
|---|---:|---:|---:|---:|---:|---:|---:|
| vanilla_moe | 0.9674 | 0.9675 | 0.9712 | 0.302 | 0.294 | 0.248 | 0.156 |
| uncertainty_moe | 0.9674 | 0.9675 | 0.9712 | 0.291 | 0.289 | 0.261 | 0.159 |
| global_umoe_regularized | 0.9574 | 0.9573 | 0.9674 | 0.167 | 0.386 | 0.035 | 0.412 |
| global_umoe_perf | 0.9649 | 0.9649 | 0.9586 | 0.002 | 0.014 | 0.949 | 0.035 |
| residual_mlp_umoe_a05 | 0.9649 | 0.9648 | 0.9662 | 0.002 | 0.635 | 0.352 | 0.011 |
| residual_mlp_umoe_a08 | 0.9524 | 0.9522 | 0.9637 | 0.000 | 0.996 | 0.003 | 0.000 |

The `uncertainty_moe` row is the main interpretable model used in the paper. It keeps strong accuracy while producing balanced view-level explanations.

### Result index

- `results/reproduction/`: visible trend vs. next-day reproduction metrics.
- `results/multiview_future1/`: pure-image multi-view strict future-label experiments.
- `results/multiview_windowtrend_ablation/`: multi-view fusion baselines and ablations.
- `results/overfit_checks/`: regularization, frozen probe, and shuffled-label sanity checks.
- `results/umoe_runs/`: U-MoE variants, histories, gate weights, and Grad-CAM examples.
- `results/umoe_metrics_summary.csv`: compact U-MoE comparison table.

## Paper Figures

The main figures are stored in `figures/`:

- `candlestick_multiview_samples.png`: representative candle/MA/RSI/MACD views;
- `task_accuracy_comparison.png`: visible-trend vs. next-day comparison;
- `gate_weights_umoe.png`: mean expert weights;
- `gradcam_umoe_aapl.png`: successful visible-trend explanation;
- `gradcam_future_failure_amd.png`: strict next-day failure case;
- `gradcam_success_failure.png`: combined success/failure visualization.

## Paper Source

The Overleaf-ready IEEE source is:

```text
paper/main_overleaf_ieee_with_figures_refs_v4.tex
```

Upload the repository folder to Overleaf and set this `.tex` file as the main document. The source expects the `figures/` directory to be available at the repository root.

## Interpretation Notes

The repository supports the following paper narrative:

1. Reproducing a chart-image trend task gives high accuracy.
2. Recasting the target as strict next-day prediction collapses performance to near chance.
3. Multi-view chart images provide complementary visual evidence for visible trend recognition.
4. U-MoE makes fusion interpretable through expert weights and expert uncertainty.
5. High entropy in strict future prediction helps diagnose when image-only evidence is unreliable.

## Citation

If you use this repository, cite the project as:

```bibtex
@misc{umoe_candlestick_2026,
  title  = {Uncertainty-aware Multi-view Expert Fusion for Interpretable Candlestick Image Trend Classification},
  author = {Anonymous},
  year   = {2026},
  note   = {Code and experiments for image-only candlestick trend classification}
}
```

## Disclaimer

This repository is for academic research only. It does not provide investment advice, trading signals, or financial recommendations. The reported next-day prediction results should be interpreted cautiously and do not imply deployable market predictability.
