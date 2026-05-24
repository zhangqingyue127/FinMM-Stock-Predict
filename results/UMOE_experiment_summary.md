# Uncertainty-Aware Multi-View Expert Fusion Summary

## Method

For each sample, four pure image views are used: candlestick, moving-average overlay, RSI, and MACD. A shared ResNet18 image encoder extracts one feature vector per view. Each view has an expert classifier:

`p_v = softmax(W_v h_v)`

Expert uncertainty is measured by predictive entropy:

`H_v = - sum_c p_v(c) log p_v(c)`

The gate produces a view score and discounts uncertain experts:

`s_v = g(h_v) - lambda H_v`

`alpha_v = softmax(s_v)`

The uncertainty-aware MoE probability is:

`p_moe = sum_v alpha_v p_v`

The residual MLP variant adds a concat-fusion branch:

`p = (1 - rho) p_moe + rho p_concat`

where `p_concat` is produced by an MLP over concatenated multi-view features. This preserves view-level gate interpretability while retaining part of the stronger concat-fusion representation.

## Main Results

| Run | Accuracy | Weighted F1 | Hit@1 | Mean Gate Weights |
|---|---:|---:|---:|---|
| Vanilla local MoE | 0.9674 | 0.9675 | 0.9712 | candle 0.302, MA 0.294, RSI 0.248, MACD 0.156 |
| Uncertainty local MoE | 0.9674 | 0.9675 | 0.9712 | candle 0.291, MA 0.289, RSI 0.261, MACD 0.159 |
| Global U-MoE regularized | 0.9574 | 0.9573 | 0.9674 | candle 0.167, MA 0.386, RSI 0.035, MACD 0.412 |
| Global U-MoE performance | 0.9649 | 0.9649 | 0.9586 | candle 0.002, MA 0.014, RSI 0.949, MACD 0.035 |
| Residual MLP U-MoE, rho=0.5 | 0.9649 | 0.9648 | 0.9662 | candle 0.002, MA 0.635, RSI 0.352, MACD 0.011 |
| Residual MLP U-MoE, rho=0.8 | 0.9524 | 0.9522 | 0.9637 | candle 0.000, MA 0.996, RSI 0.003, MACD 0.000 |

## Interpretation

The strongest U-MoE result is the local uncertainty-aware model, which reaches 96.74% accuracy and retains balanced gate weights across views. The residual MLP variant is useful as a formulaic extension, but a high residual weight can cause gate collapse.

The best overall accuracy remains the earlier concat-fusion baseline at 98.12%. Therefore, this method should be positioned as an interpretable uncertainty-aware fusion mechanism, not as a pure accuracy improvement over concat fusion.

The most defensible paper claim is: static concat fusion is the strongest classifier, while U-MoE explains which technical image views dominate each decision and exposes uncertainty-driven view reliability.
