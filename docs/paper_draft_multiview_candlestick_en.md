# Multi-view Candlestick Image Representation Learning for Interpretable Stock Trend Classification

## Abstract

Candlestick-chart images have recently been used as visual representations for stock trend classification, and several studies report high predictive accuracy from chart-based deep learning models. However, the experimental meaning of such accuracy is often ambiguous: a model may learn to recognize the trend already visible within the input window rather than predict future price movement beyond that window. This paper revisits image-based stock trend classification under a leakage-aware evaluation protocol and proposes a pure-image multi-view representation learning framework for interpretable trend analysis. We first reproduce a representative candlestick-image classification setting using ten U.S. stocks from 20 February 2020 to 18 December 2023, 30-day chart windows, 5,283 samples, and strict chronological train/validation/test splits. Under a paper-style window-trend label, a lightweight CNN achieves 98.37% test accuracy, whereas under a strict next-day label the same setting drops to 49.62%, suggesting that the high score mainly reflects visible trend recognition rather than genuine future return prediction.

Building on this observation, we introduce a multi-view candlestick image representation framework that renders each stock window into four technical image views: raw candlestick, moving-average candlestick, RSI, and MACD. A ResNet18 encoder is pretrained using a multi-view consistency contrastive objective, where different views of the same temporal window are treated as positive pairs. The learned representation is then evaluated for trend classification, similar-image retrieval, and Grad-CAM-based interpretability. Feature-level concatenation of independently encoded views achieves 98.12% accuracy and 98.13% weighted F1 on the window-trend task, approaching the reproduced single-chart CNN while additionally supporting cross-view retrieval and interpretability analysis. A regularized variant with augmentation, label smoothing, stronger dropout, and weight decay retains 97.12% accuracy. A frozen-encoder linear probe achieves 93.98%, and a shuffled-label sanity check substantially degrades performance, supporting that the representation captures trend-related visual structures rather than merely memorizing artifacts. Overall, our findings indicate that candlestick-image models can effectively identify and retrieve visible trend patterns, but strict future prediction from static chart images remains weak. The proposed framework provides a transparent and leakage-aware evaluation protocol for image-based financial time-series research.

**Keywords:** candlestick charts, financial time series, image representation learning, contrastive learning, multi-view learning, explainable AI, Grad-CAM

## 1. Introduction

Financial time-series forecasting has long attracted attention from machine learning, data mining, and computational finance communities. In addition to tabular technical indicators and textual market information, chart-based representations have become increasingly popular because they transform numerical price sequences into visual patterns that can be processed by convolutional neural networks and other computer vision models. Candlestick charts are especially attractive: they encode open, high, low, and close prices in a compact visual form and are widely used by market participants to reason about short-term price dynamics.

Recent image-based stock prediction studies report promising results by converting price histories into candlestick-chart images and training deep neural networks for trend classification. Nevertheless, there is a subtle but important ambiguity in this literature. A high trend-classification accuracy does not necessarily imply that a model predicts future returns. If the target label is derived from the price movement already contained in the input chart window, the model is solving a visual trend recognition task: it recognizes whether the displayed window itself trends upward or downward. This task is useful for representation learning and pattern retrieval, but it is conceptually different from predicting whether the price will move up or down after the window.

This distinction is not merely semantic. In financial machine learning, overly optimistic results often arise from leakage, overlapping windows, random splits, or target definitions that encode information already visible to the model. Therefore, chart-based financial models should be evaluated with explicit target definitions and chronological splits. Without such care, a visually impressive accuracy may be incorrectly interpreted as evidence of predictive power.

This paper revisits candlestick-image stock trend classification from a leakage-aware perspective. We first reproduce a representative image-based candlestick classification setting using ten U.S. stocks, 30-day chart windows, and a lightweight convolutional neural network. We compare two labels under the same data and chronological split: a **window-trend** label that indicates whether the closing price at the end of the input window is higher than at the beginning, and a **future-k** label that indicates whether the price increases after the window. The results are sharply different. The window-trend task reaches 98.37% test accuracy, whereas the strict next-day task is near random. This suggests that the high-performance setting primarily captures visible trend recognition, not robust future prediction.

Rather than treating this as a failure, we argue that it provides a more rigorous foundation for image-based financial representation learning. Candlestick images may not reliably predict next-day returns by themselves, but they can still encode valuable visual structures for trend recognition, retrieval, and interpretation. We therefore propose a pure-image multi-view candlestick representation learning framework. Each 30-day window is rendered into four image views: raw candlestick, candlestick with moving averages, RSI, and MACD. A shared ResNet18 encoder is pretrained using a multi-view consistency objective, where different views from the same stock window serve as positive pairs. The representation is then used for downstream classification, image retrieval, and Grad-CAM explanation.

The goal of this paper is not to overclaim market predictability. Instead, we aim to clarify what chart-image models can and cannot learn. We show that the proposed multi-view representation performs strongly on visible trend recognition and similar-image retrieval, while strict next-day prediction remains weak. We further provide overfitting diagnostics, including a regularized model, a frozen-encoder probe, ticker-level testing, and shuffled-label sanity checks.

The main contributions are:

1. We clarify the distinction between visible window-trend recognition and future return prediction in candlestick-image classification.
2. We reproduce a representative high-accuracy chart-classification setting and show that the result collapses under a strict next-day target.
3. We propose a pure-image multi-view candlestick representation learning framework based on candle, moving-average, RSI, and MACD image views.
4. We introduce a multi-view consistency pretraining strategy and show that feature-level concatenation of independently encoded views substantially improves over mean fusion.
5. We evaluate classification, retrieval, Grad-CAM interpretability, and overfitting diagnostics under chronological splits.

## 2. Related Work

### 2.1 Financial Time-Series Prediction

Machine learning methods for financial time-series prediction include classical statistical models, tree-based models, recurrent neural networks, convolutional models, attention mechanisms, and transformer architectures [REF]. These methods typically operate on numerical features such as historical returns, volatility, volume, or technical indicators. However, financial prediction remains difficult because price series are noisy, non-stationary, and affected by latent market mechanisms. Consequently, evaluation design is critical: chronological splits, leakage control, and robust out-of-sample testing are essential for assessing whether a model learns predictive structure rather than artifacts.

### 2.2 Candlestick Chart Image Classification

Candlestick charts provide a visual encoding of price movement and are widely used in technical analysis. Several recent studies convert candlestick charts into RGB images and apply object recognition or convolutional neural networks for stock pattern recognition and trend classification [REF]. Some papers report high accuracy on trend classification tasks. However, many such settings require careful inspection because the target may correspond to the trend already visible in the chart. This paper revisits that issue empirically by comparing window-trend and future-return labels under the same data split.

### 2.3 Multi-view and Self-supervised Representation Learning

Self-supervised learning has become an effective approach for learning visual representations without relying solely on labels. Contrastive learning methods such as SimCLR learn representations by pulling together augmented views of the same instance and pushing apart different instances [REF]. Multi-view learning extends this idea by exploiting complementary observations of the same underlying object or event. In financial chart analysis, different technical indicators can be interpreted as different visual views of the same temporal window. This motivates our use of candle, moving-average, RSI, and MACD images as positive views in a contrastive pretraining objective.

### 2.4 Explainable AI for Financial Models

Interpretability is important in financial applications because model decisions may influence investment or risk decisions. For image models, Grad-CAM provides a widely used way to visualize discriminative regions by backpropagating class-specific gradients to convolutional feature maps [REF]. In candlestick-image analysis, Grad-CAM can help identify whether the model attends to meaningful chart regions, such as trend slopes, moving-average crossings, RSI levels, or MACD changes, rather than irrelevant image borders or blank backgrounds.

## 3. Problem Definition and Evaluation Protocol

### 3.1 Data Setting

We follow a representative candlestick-image classification setting using ten U.S. stocks: AAPL, TSLA, MSFT, AMZN, NVDA, META, GOOG, JPM, AMD, and BAC. Daily OHLC price data cover the period from 20 February 2020 to 18 December 2023. Each sample is constructed from a 30-trading-day window. The final dataset contains 5,283 image samples and is split chronologically into 3,667 training samples, 818 validation samples, and 798 test samples.

The chronological split is used throughout the paper. This design avoids random assignment of highly overlapping time windows across training and test sets, which could otherwise lead to inflated performance.

### 3.2 Label Definitions

We distinguish two target definitions.

The **window-trend** label is defined as:

```text
y = 1 if Close_t > Close_{t-29}, else 0,
```

where the 30-day input window spans from \(t-29\) to \(t\). This label describes the trend visible inside the input image.

The **future-k** label is defined as:

```text
y = 1 if Close_{t+k} > Close_t, else 0.
```

For \(k=1\), the task is strict next-day direction prediction. This label is not directly visible in the input window and therefore better reflects future prediction.

These two tasks should not be conflated. The window-trend task evaluates visual pattern recognition, whereas the future-k task evaluates predictive ability beyond the input window.

### 3.3 Evaluation Metrics

We report accuracy, weighted precision, weighted recall, weighted F1, confusion matrices, and class-level classification reports. For representation quality, we also compute similar-image retrieval metrics using learned embeddings, including Hit@1, Hit@5, and Hit@10. For interpretability, we provide Grad-CAM visualizations across multiple views.

## 4. Method

### 4.1 Single-View Reproduction Model

The reproduction baseline follows a lightweight CNN architecture commonly used in candlestick-image classification. Each 30-day chart is rendered as a 100 by 100 RGB image containing candlesticks and technical subpanels. The CNN contains three convolutional blocks with 32, 64, and 128 filters, each followed by batch normalization, LeakyReLU activation, and max pooling. Dropout and a dense layer are used before the final two-class softmax classifier. The model is trained with Adam, a learning rate of \(10^{-3}\), weight decay of \(10^{-4}\), batch size 32, and early stopping.

This baseline is used to reproduce the paper-style high-accuracy window-trend classification result and to compare it with strict next-day prediction.

### 4.2 Multi-view Image Construction

For each 30-day price window, we generate four pure-image views:

1. **Candlestick view:** raw OHLC candlestick chart.
2. **Moving-average view:** candlestick chart with moving average overlays.
3. **RSI view:** relative strength index line image with overbought and oversold reference levels.
4. **MACD view:** MACD histogram and signal-line image.

No raw tabular features, textual information, or news data are provided to the model. Technical information is introduced only through rendered image views.

### 4.3 Multi-view Consistency Pretraining

Let \(x_i^v\) denote view \(v\) of sample \(i\). A shared ResNet18 encoder \(f_\theta(\cdot)\) maps each image view to a latent representation, followed by a projection head \(g_\phi(\cdot)\). During pretraining, two different views of the same time window are sampled as a positive pair, while views from other windows in the mini-batch act as negatives.

The contrastive loss encourages representations of the same window across different technical views to be close:

```text
z_i^v = normalize(g_phi(f_theta(x_i^v))).
```

For a positive pair \((z_i^a, z_i^b)\), we use a temperature-scaled cross-entropy contrastive objective similar to NT-Xent [REF]. This pretraining encourages the encoder to learn view-invariant visual structures associated with the same underlying price window.

### 4.4 Downstream Multi-view Classification

After pretraining, each view is encoded independently by the shared ResNet18 encoder. We evaluate two fusion strategies:

1. **Mean fusion:** average the four view embeddings.
2. **Concat fusion:** concatenate the four view embeddings and feed them to a classification head.

Empirically, concat fusion performs substantially better because it preserves view-specific information. The final classification head consists of a linear layer, ReLU activation, dropout, and a two-class output layer.

### 4.5 Retrieval and Interpretability

For retrieval, we use the learned downstream feature vectors and cosine similarity to rank test samples. A retrieval is counted as successful at rank \(k\) if at least one of the top-\(k\) retrieved samples has the same class label.

For interpretability, we compute Grad-CAM heatmaps for each view. In the concat-fusion model, all four views are forwarded jointly, and class-specific gradients are backpropagated to the final convolutional block of the shared encoder. This yields view-specific heatmaps indicating which regions contribute to the prediction.

## 5. Experiments

### 5.1 Reproduction Results

Table 1 compares the paper-style window-trend task and the strict next-day prediction task using the lightweight CNN baseline.

**Table 1. Reproduction under different label definitions.**

| Task | Label | Model | Test Accuracy | Weighted F1 |
| --- | --- | --- | ---: | ---: |
| Paper-style trend classification | window-trend | Lightweight CNN | 0.9837 | approximately 0.9837 |
| Strict future prediction | next-day | Lightweight CNN | 0.4962 | approximately 0.4757 |

The same model and data setting yield dramatically different results depending on the label definition. The paper-style task is highly accurate because the label is visually encoded in the input window. In contrast, next-day direction prediction is near random. This confirms that high accuracy in candlestick-image trend classification should not automatically be interpreted as future prediction ability.

### 5.2 Multi-view Future Prediction

We next evaluate the multi-view model under a strict future label with \(k=1\). Table 2 shows that future prediction remains weak even with multi-view representation learning.

**Table 2. Multi-view strict future prediction.**

| Model | Label | Test Accuracy | Weighted F1 | Retrieval Hit@1 |
| --- | --- | ---: | ---: | ---: |
| ImageNet ResNet18 multi-view classifier | future-k=1 | 0.4461 | 0.3278 | 0.5000 |
| Multi-view pretraining + classifier | future-k=1 | 0.4850 | 0.4300 | 0.4975 |

Multi-view consistency pretraining slightly improves over the ImageNet baseline in classification F1, but the result remains close to random. This supports the conclusion that static chart images alone provide limited evidence for next-day direction prediction in this setting.

### 5.3 Multi-view Window-Trend Recognition

We then evaluate multi-view representation learning on the window-trend task. Table 3 summarizes the results.

**Table 3. Multi-view window-trend classification and ablations.**

| Model | Test Accuracy | Weighted F1 | Retrieval Hit@1 |
| --- | ---: | ---: | ---: |
| Mean fusion | 0.9561 | 0.9562 | 0.9574 |
| Concat fusion | 0.9812 | 0.9813 | 0.9674 |
| 2x2 composite image ResNet18 | 0.9637 | 0.9638 | 0.9586 |
| Five-view concat with full chart | 0.9662 | 0.9662 | 0.9662 |
| Probability ensemble of concat models | 0.9787 | 0.9787 | not reported |

The concat-fusion model is the strongest multi-view architecture. Mean fusion loses view-specific information, while 2x2 composite images force heterogeneous views into a single image canvas and reduce detail. Adding an additional full-chart view does not improve performance, likely because it introduces redundant information and increases overfitting risk.

### 5.4 Robustness and Overfitting Diagnostics

Because the window-trend task yields high accuracy, we perform additional diagnostics to assess overfitting.

**Table 4. Overfitting diagnostics.**

| Diagnostic | Setting | Test Accuracy | Weighted F1 | Retrieval Hit@1 |
| --- | --- | ---: | ---: | ---: |
| Best-performance model | concat fusion | 0.9812 | 0.9813 | 0.9674 |
| Regularized model | augmentation + label smoothing + dropout + stronger weight decay | 0.9712 | 0.9712 | 0.9599 |
| Frozen-encoder probe | freeze encoder, train classifier only | 0.9398 | 0.9400 | 0.9273 |
| Shuffled training labels | train labels randomly permuted | 0.6992 | 0.6516 | 0.9261 |

The regularized model sacrifices about one percentage point of accuracy but remains strong. This suggests that the model does not rely solely on fragile memorization. The frozen-encoder probe is particularly informative: even without updating the ResNet encoder during downstream training, the classifier reaches 93.98% test accuracy, indicating that the pretrained representation itself contains trend-related visual information.

The shuffled-label sanity check further reduces the likelihood of leakage. If the model were exploiting file paths, image layout, dates, or other non-label artifacts, shuffling training labels might still preserve high validation and test scores. Instead, performance drops substantially and predictions collapse toward the majority class. This suggests that the high score is not caused by trivial path or layout leakage.

### 5.5 Ticker-Level Stability

The regularized model maintains stable performance across all ten stocks. Ticker-level test accuracy ranges from 93.67% to 100.00%. This reduces the concern that aggregate performance is driven by a small subset of easy stocks.

**Table 5. Ticker-level performance of the regularized model.**

| Ticker | Test Accuracy |
| --- | ---: |
| AAPL | 0.9634 |
| AMD | 0.9506 |
| AMZN | 0.9875 |
| BAC | 0.9780 |
| GOOG | 0.9882 |
| JPM | 0.9875 |
| META | 0.9367 |
| MSFT | 1.0000 |
| NVDA | 0.9452 |
| TSLA | 0.9697 |

## 6. Interpretability Analysis

We use Grad-CAM to inspect whether the regularized multi-view model attends to meaningful visual regions. For the candlestick and moving-average views, heatmaps should ideally concentrate on price movement segments, candle bodies, trend slopes, and moving-average turning points. For RSI and MACD views, meaningful attention should occur near high/low RSI regions, oscillator changes, MACD histogram transitions, and signal-line interactions.

The regularized model's Grad-CAM examples show attention distributed over technical chart regions rather than consistently focusing on image borders, blank backgrounds, or fixed corners. This does not prove causal financial reasoning, but it provides qualitative evidence that the model uses chart-relevant visual structures. In the paper, we recommend presenting both correct high-confidence examples and failure cases. Failure cases are important because they reveal when similar visual trends become ambiguous or when indicator views disagree.

Grad-CAM should be interpreted together with quantitative diagnostics. The combination of regularized training, frozen-encoder probing, ticker-level stability, shuffled-label sanity checks, and view-specific heatmaps forms a stronger argument than any single explanation method alone.

## 7. Discussion

### 7.1 What the Model Learns

The experiments suggest that candlestick-image models can learn visible trend structures very effectively. When the target label describes the direction already shown in the input window, both single-view CNNs and multi-view ResNet models achieve high accuracy. Multi-view learning is especially useful for representation analysis because it separates complementary technical views and allows view-specific interpretation.

### 7.2 What the Model Does Not Learn

The same models fail to predict strict next-day direction. This finding is important. It indicates that high accuracy in image-based trend classification should not be equated with profitable or reliable future forecasting. The visual chart contains strong information about past and current trend states, but that does not necessarily imply predictive information about the next trading day.

### 7.3 Why Multi-view Learning Still Matters

Even if future prediction remains weak, multi-view representation learning is still valuable. It provides a structured way to learn visual representations of financial time windows, retrieve historically similar chart patterns, and interpret technical indicators in image space. Such representations may support downstream tasks such as chart pattern search, market regime clustering, decision-support visualization, or longer-horizon studies where labels are less noisy than next-day direction.

### 7.4 Implications for Financial AI Evaluation

This study highlights the importance of label design. A model that recognizes an upward chart window is not necessarily a model that predicts future upward movement. Future work in image-based financial AI should explicitly report whether labels are inside-window or out-of-window, whether splits are chronological, and whether overlapping windows cross split boundaries.

## 8. Limitations

This paper has several limitations. First, the dataset contains ten U.S. stocks and one historical period. Broader evaluation across markets, sectors, and regimes is needed. Second, the strict future-prediction task is tested primarily with \(k=1\). Longer horizons, such as 5-day or 10-day returns, may be more compatible with chart patterns and should be evaluated. Third, the image generation design may influence performance; alternative chart styles, resolutions, or indicator panels could affect both accuracy and interpretability. Fourth, Grad-CAM offers qualitative explanations but cannot fully establish causal decision mechanisms. Finally, the current study focuses on pure-image modality; it does not evaluate whether combining images with carefully controlled numerical or textual information improves future prediction.

## 9. Conclusion

This paper revisits image-based candlestick stock trend classification under a leakage-aware evaluation framework. We show that a high-accuracy paper-style window-trend classifier can be reproduced, but that strict next-day prediction collapses to near-random accuracy under the same chronological split. This demonstrates the importance of distinguishing visible trend recognition from future prediction.

We propose a pure-image multi-view candlestick representation learning framework based on candle, moving-average, RSI, and MACD image views. Multi-view consistency pretraining and feature-level concat fusion achieve strong window-trend recognition, high retrieval performance, and interpretable Grad-CAM visualizations. Robustness checks, including regularization, frozen-encoder probing, ticker-level analysis, and shuffled-label sanity testing, suggest that the model captures chart-relevant visual structures rather than only memorizing artifacts.

The central conclusion is deliberately cautious: candlestick-image models are effective at recognizing and retrieving visible trend patterns, but static chart images alone provide weak evidence for strict next-day prediction. The proposed framework offers a transparent evaluation protocol for future research in image-based financial time-series representation learning.

## References

Formal references should be inserted here using the target journal's citation style. Suggested citation groups include:

1. Candlestick chart image classification and object recognition in financial markets.
2. Image-based time-series trend classification using deep learning.
3. Financial time-series prediction surveys.
4. Contrastive learning methods such as SimCLR.
5. Multi-view representation learning.
6. Grad-CAM and explainable AI methods.
7. Leakage-aware evaluation and backtesting methodology in financial machine learning.

