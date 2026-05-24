# Uncertainty-aware Multi-view Expert Fusion for Interpretable Candlestick Image Trend Classification

## Abstract

Candlestick-chart images provide a visually intuitive representation of financial time series and have been increasingly used for stock trend classification. However, high accuracy in chart-image classification is often difficult to interpret: a model may recognize the trend already visible inside the input window rather than predict the subsequent price movement, and standard feature-fusion models provide limited insight into which technical views support a decision. This paper studies candlestick image trend classification from the perspective of interpretable multi-expert prediction. We propose an **Uncertainty-aware Multi-view Expert Fusion** framework, termed **U-MoE**, that treats different technical chart renderings as view-specific image experts. Each stock window is rendered into four pure image views: raw candlestick, moving-average candlestick, RSI, and MACD. A shared ResNet18 encoder extracts view-level visual representations, each expert produces a class distribution, and a gating module dynamically fuses expert predictions. Crucially, the gate is regularized by expert predictive entropy, allowing uncertain views to receive lower decision weight.

We evaluate U-MoE under strict chronological splits on 5,283 samples constructed from ten U.S. stocks. Our experiments show a clear distinction between visible trend recognition and future direction prediction. A reproduced paper-style window-trend classifier reaches 98.37% test accuracy, whereas strict next-day prediction remains near random. On the window-trend task, black-box concat fusion achieves the highest raw accuracy of 98.12%, while U-MoE achieves 96.74% accuracy with explicit view-level expert weights and uncertainty estimates. On next-day prediction, U-MoE does not improve accuracy, but its expert entropies approach the maximum binary entropy, revealing that the model cannot identify reliable visual evidence for the future label. These results support a cautious but useful conclusion: candlestick images are effective for recognizing visible trend structures, while U-MoE provides an interpretable mechanism for diagnosing which chart views contribute to a prediction and when the model should be considered uncertain.

**Keywords:** candlestick chart, financial image analysis, mixture of experts, uncertainty estimation, interpretable machine learning, multi-view learning, Grad-CAM

## 1. Introduction

Financial time-series prediction is a challenging problem because asset prices are noisy, non-stationary, and shaped by latent market mechanisms that are only partially observable. In recent years, an alternative to numerical feature engineering has gained attention: converting historical price windows into candlestick-chart images and applying computer vision models to classify future or contemporaneous market trends. This line of work is appealing because candlestick charts encode open, high, low, and close prices in a compact visual format familiar to market participants, and convolutional neural networks can learn local visual patterns from such images.

Despite promising reported results, image-based stock prediction raises two important questions. First, what exactly is being predicted? In many chart-image settings, the label describes whether the displayed window itself moves upward or downward. A model can then obtain high accuracy by recognizing a trend already visible inside the input image. This is a legitimate visual trend recognition task, but it differs from predicting the next trading day's price direction. Second, when a model uses multiple technical chart views, such as candlesticks, moving averages, RSI, and MACD, how can we know which view contributes to the final decision? Standard feature concatenation can be accurate, but it provides little view-level interpretability.

This paper focuses on the second issue while explicitly controlling for the first. We aim to build an interpretable pure-image model for candlestick trend classification, not to overstate the predictability of short-term returns. Our central hypothesis is that different technical image views behave like distinct experts: a candlestick view may capture price body and wick patterns, a moving-average view may capture trend direction, RSI may capture momentum saturation, and MACD may capture oscillator changes. A mixture-of-experts architecture is therefore a natural fit for multi-view financial chart images, provided that the model can also quantify when an expert is uncertain.

We propose **Uncertainty-aware Multi-view Expert Fusion** (**U-MoE**), a multi-view image model in which each chart view has an expert classifier and a gate dynamically aggregates expert predictions. Unlike ordinary MoE models that rely only on learned gate scores, U-MoE discounts experts with high predictive entropy. This design produces not only a final class probability, but also interpretable quantities: expert probabilities, expert entropies, and gate weights. These quantities allow us to ask whether the model relies more on candlestick, moving-average, RSI, or MACD views for a given prediction, and whether the decision is supported by confident or uncertain experts.

Our experiments are deliberately structured around two label definitions. The **window-trend** label indicates whether the closing price at the end of a 30-day input window is higher than at the start of the same window. This label is visually encoded in the chart and measures trend recognition. The **next-day** label indicates whether the closing price on the next trading day is higher than the current closing price. This label measures strict future direction prediction. Under the same chronological split, these two tasks behave very differently. Window-trend classification reaches high accuracy, whereas next-day prediction remains near random across CNN, concat-fusion, and U-MoE models.

The main contributions of this paper are:

1. We formulate a pure-image multi-view candlestick trend classification setting in which candle, moving-average, RSI, and MACD charts are treated as view-specific experts.
2. We propose U-MoE, an uncertainty-aware mixture-of-experts fusion method that combines expert predictions using entropy-regularized view weights.
3. We show that U-MoE provides explicit view-level interpretability through expert weights and predictive entropies, while retaining strong trend-recognition performance.
4. We empirically distinguish visible window-trend recognition from strict next-day prediction and show that U-MoE can diagnose prediction failure through high expert uncertainty.
5. We complement quantitative results with retrieval, shuffled-label sanity checks, frozen-probe diagnostics, and Grad-CAM visualizations.

## 2. Related Work

### 2.1 Candlestick Image Classification

Candlestick charts are widely used in technical analysis because they summarize price action through visual primitives such as candle bodies, shadows, gaps, and trend slopes. Several recent studies transform financial time-series windows into chart images and apply convolutional neural networks, object recognition methods, or image classification architectures to identify trend or pattern classes. These methods demonstrate that visual encodings can be useful for learning chart structures. However, reported high accuracies often depend strongly on label definition, window construction, and split protocol. If the target label is derived from the same window displayed in the input image, the task is trend recognition rather than future prediction.

### 2.2 Multi-view Financial Representations

Financial market states can be represented from multiple perspectives: raw price charts, moving-average overlays, momentum oscillators, volatility indicators, and volume patterns. Multi-view learning provides a principled way to combine such complementary observations. In this paper, we restrict all information to rendered images, avoiding numerical tabular features or text modalities. This allows us to study whether a pure image model can learn and explain trend-related visual structures from multiple technical chart views.

### 2.3 Mixture of Experts

Mixture-of-experts models decompose prediction into multiple specialized experts and a gating network that assigns input-dependent expert weights. MoE architectures are useful when different subsets of features, tasks, or data regimes require different decision rules. In financial chart analysis, different technical indicators may be informative under different market conditions. This motivates a view-level MoE design in which candlestick, MA, RSI, and MACD images are separate experts. Compared with feature concatenation, MoE provides a natural interpretation: the gate weight indicates how much the model relies on each expert.

### 2.4 Uncertainty and Interpretability

Uncertainty estimation is important in financial applications because a model's abstention or lack of confidence can be as informative as its predicted class. Predictive entropy is a simple and widely used measure of classification uncertainty. For a two-class prediction, entropy approaches its maximum when the model assigns nearly equal probability to both classes. We use expert entropy to regularize view fusion and to diagnose whether individual chart views provide reliable evidence. For visual explanation, we also use Grad-CAM to inspect which image regions contribute to predictions.

## 3. Problem Definition

### 3.1 Data Setting

We use daily OHLC data from ten U.S. stocks: AAPL, TSLA, MSFT, AMZN, NVDA, META, GOOG, JPM, AMD, and BAC. The historical period spans from 20 February 2020 to 18 December 2023. Each sample is constructed from a 30-trading-day window. The dataset contains 5,283 samples split chronologically into 3,667 training samples, 818 validation samples, and 798 test samples.

The chronological split is essential. Random splits can place highly overlapping windows from nearby dates into both training and test sets, producing overly optimistic results. Throughout this paper, all reported test results use a future chronological holdout.

### 3.2 Label Definitions

We consider two labels.

The **window-trend** label is

\[
y_i^{win} = \mathbb{1}(C_t > C_{t-29}),
\]

where \(C_t\) is the closing price at the end of the input window and \(C_{t-29}\) is the closing price at the start of the same 30-day window. This label measures whether the input chart itself displays an upward or downward trend.

The **next-day** label is

\[
y_i^{next} = \mathbb{1}(C_{t+1} > C_t).
\]

This label is outside the input window and corresponds to strict next-trading-day direction prediction.

We treat these as separate tasks. Window-trend classification evaluates visual trend recognition. Next-day classification evaluates whether the image contains predictive information for the immediate future.

### 3.3 Multi-view Image Inputs

For each window, we render four RGB image views:

- \(x^{candle}\): raw candlestick chart.
- \(x^{MA}\): candlestick chart with moving-average overlays.
- \(x^{RSI}\): relative strength index chart.
- \(x^{MACD}\): MACD histogram and signal-line chart.

No numerical OHLC values, volume features, text, news, or tabular technical indicators are provided to the model. All technical information is visible only through images.

## 4. Method

### 4.1 Shared Visual Encoder

Let \(V=\{1,\ldots,M\}\) denote the set of views, with \(M=4\). For sample \(i\), view \(v\) is an image \(x_i^v\). A shared ResNet18 encoder \(f_\theta\) maps each view to a feature vector:

\[
h_i^v = f_\theta(x_i^v) \in \mathbb{R}^d.
\]

The encoder is initialized from ImageNet weights and can optionally be initialized from a multi-view consistency pretraining checkpoint. The same encoder is shared across views to encourage a common visual representation space.

### 4.2 View-specific Experts

Each view has an expert classifier \(E_v\). The expert produces logits and probabilities:

\[
z_i^v = W_v h_i^v + b_v,
\]

\[
p_i^v = \mathrm{softmax}(z_i^v).
\]

The expert prediction \(p_i^v\) can be interpreted as the class belief of a specific technical chart view.

### 4.3 Expert Uncertainty

For each expert, we compute predictive entropy:

\[
H_i^v = - \sum_{c=1}^{K} p_i^v(c)\log p_i^v(c),
\]

where \(K=2\) for binary trend classification. Entropy is high when an expert assigns similar probabilities to upward and downward classes. In binary classification, the maximum entropy is \(\log 2 \approx 0.693\).

### 4.4 Uncertainty-aware Gating

The gate produces an input-dependent score for each expert. In the local-gate variant, the score is computed from each view feature:

\[
a_i^v = g_\psi(h_i^v).
\]

In the uncertainty-aware version, the score is penalized by expert entropy:

\[
s_i^v = a_i^v - \lambda H_i^v,
\]

where \(\lambda\) controls the strength of uncertainty discounting. Gate weights are obtained by a softmax over views:

\[
\alpha_i^v = \frac{\exp(s_i^v)}{\sum_{u=1}^{M}\exp(s_i^u)}.
\]

The final U-MoE prediction is a weighted average of expert probabilities:

\[
p_i^{MoE} = \sum_{v=1}^{M} \alpha_i^v p_i^v.
\]

This formulation makes the prediction decomposable. The final probability is determined by expert beliefs \(p_i^v\), expert uncertainty \(H_i^v\), and gate weights \(\alpha_i^v\).

### 4.5 Residual Concat Branch

We also evaluate a residual version that combines U-MoE with a standard concat-fusion classifier:

\[
h_i^{cat} = [h_i^1; h_i^2; \ldots; h_i^M],
\]

\[
p_i^{cat} = \mathrm{softmax}(q_\omega(h_i^{cat})).
\]

The residual prediction is

\[
p_i = (1-\rho)p_i^{MoE} + \rho p_i^{cat}.
\]

This variant tests whether a black-box high-capacity concat branch can improve performance while preserving expert weights for interpretation. Empirically, however, high residual weights can cause the gate to collapse to a single dominant view, so we treat this variant as an ablation rather than the main method.

### 4.6 Training Objective

The main classification loss is the negative log-likelihood of the fused prediction:

\[
\mathcal{L}_{cls} = - \log p_i(y_i).
\]

We also include an auxiliary expert loss:

\[
\mathcal{L}_{aux} = \frac{1}{M}\sum_{v=1}^{M} CE(z_i^v, y_i),
\]

which encourages each expert to remain individually predictive. A consistency term aligns expert distributions with the fused prediction:

\[
\mathcal{L}_{cons} = \frac{1}{M}\sum_{v=1}^{M} KL(p_i \parallel p_i^v).
\]

The total loss is

\[
\mathcal{L} = \mathcal{L}_{cls} + \beta\mathcal{L}_{aux} + \eta\mathcal{L}_{cons} + \gamma\mathcal{L}_{div}.
\]

\(\mathcal{L}_{div}\) is an optional feature-diversity regularizer. We use early stopping on validation loss.

## 5. Experimental Setup

### 5.1 Baselines

We compare U-MoE with:

- **Lightweight CNN:** a reproduction baseline for single candlestick-image classification.
- **Mean fusion:** average of four view embeddings followed by a classifier.
- **Concat fusion:** concatenation of four view embeddings followed by an MLP classifier.
- **Regularized concat:** concat fusion with stronger dropout, data augmentation, label smoothing, and weight decay.
- **Frozen encoder probe:** encoder frozen, classifier trained on downstream labels.
- **Shuffled-label sanity check:** training labels randomly permuted while validation and test labels remain true.

Concat fusion is the strongest accuracy baseline but is not our proposed contribution. It is included to test whether interpretability comes at a performance cost.

### 5.2 Metrics

We report accuracy, weighted F1, confusion matrices, and class-level reports. For representation quality, we report retrieval Hit@1 using cosine similarity between learned features. For interpretability, we analyze expert gate weights, expert entropies, and Grad-CAM heatmaps.

## 6. Results

### 6.1 Window-trend Recognition vs Next-day Prediction

Table 1 shows the effect of label definition.

**Table 1. Reproduction under different target definitions.**

| Task | Label | Model | Test Accuracy | Weighted F1 |
|---|---|---|---:|---:|
| Visible trend recognition | window-trend | Lightweight CNN | 0.9837 | approximately 0.9837 |
| Strict future prediction | next-day | Lightweight CNN | 0.4962 | approximately 0.4757 |

The same image-based setting yields very different results depending on the target. The high window-trend score indicates that the model can recognize the trend already displayed in the image. The next-day score is near random, suggesting that static chart images alone do not provide reliable immediate future-direction signals in this setting.

### 6.2 Multi-view Fusion Baselines

Table 2 compares multi-view fusion strategies on the window-trend task.

**Table 2. Multi-view window-trend classification.**

| Model | Test Accuracy | Weighted F1 | Retrieval Hit@1 |
|---|---:|---:|---:|
| Mean fusion | 0.9561 | 0.9562 | 0.9574 |
| Concat fusion | 0.9812 | 0.9813 | 0.9674 |
| Regularized concat | 0.9712 | 0.9712 | 0.9599 |
| Frozen encoder probe | 0.9398 | 0.9400 | approximately 0.9273 |

Concat fusion achieves the best raw accuracy, confirming that view-specific information is complementary. However, concat fusion does not explicitly reveal which view contributes to a decision. U-MoE addresses this interpretability gap.

### 6.3 U-MoE on Window-trend Classification

Table 3 presents U-MoE results on the window-trend task.

**Table 3. U-MoE window-trend results.**

| Model | Test Accuracy | Weighted F1 | Hit@1 | Mean Gate Weights |
|---|---:|---:|---:|---|
| Vanilla local MoE | 0.9674 | 0.9675 | 0.9712 | candle 0.302, MA 0.294, RSI 0.248, MACD 0.156 |
| Uncertainty local MoE | 0.9674 | 0.9675 | 0.9712 | candle 0.291, MA 0.289, RSI 0.261, MACD 0.159 |
| Global U-MoE regularized | 0.9574 | 0.9573 | 0.9674 | candle 0.167, MA 0.386, RSI 0.035, MACD 0.412 |
| Residual MLP U-MoE, \(\rho=0.5\) | 0.9649 | 0.9648 | 0.9662 | candle 0.002, MA 0.635, RSI 0.352, MACD 0.011 |

The local uncertainty-aware U-MoE achieves 96.74% accuracy with balanced expert weights. Although it does not exceed concat fusion in accuracy, it provides a direct decomposition of model behavior. The model assigns substantial weight to candlestick, MA, and RSI views, while MACD receives lower average weight. This suggests that, for visible trend recognition, price structure and trend/momentum views are more consistently informative than MACD in the current setup.

The global and residual variants show that more flexible gates can become less stable. Some variants collapse to a dominant view, such as RSI or MA. This gate collapse is an important diagnostic: maximizing accuracy alone may reduce interpretability and view diversity.

### 6.4 U-MoE on Next-day Prediction

Table 4 evaluates whether U-MoE improves strict next-day prediction.

**Table 4. Next-day prediction results.**

| Model | Label | Test Accuracy | Weighted F1 | Retrieval Hit@1 |
|---|---|---:|---:|---:|
| Lightweight CNN | next-day | 0.4962 | approximately 0.4757 | not reported |
| Multi-view pretraining + classifier | future-k=1 | 0.4850 | 0.4300 | 0.4975 |
| U-MoE local uncertainty | future-k=1 | 0.4724 | 0.4732 | 0.4987 |
| Residual MLP U-MoE | future-k=1 | 0.4712 | 0.4305 | 0.4987 |

U-MoE does not improve next-day prediction. This negative result is important. A more expressive interpretable fusion model does not overcome the lack of stable predictive signal in the image-only input.

The uncertainty estimates further support this interpretation. For local U-MoE, mean expert entropies are:

| Expert | Mean Entropy |
|---|---:|
| candle | 0.6789 |
| MA | 0.6797 |
| RSI | 0.6734 |
| MACD | 0.6825 |

Since the maximum binary entropy is approximately 0.693, all experts are highly uncertain. Thus, U-MoE explains the failure mode: the model is not merely choosing the wrong view; rather, all views provide weak and uncertain evidence for next-day direction.

### 6.5 Shuffled-label Sanity Check

Table 5 reports the shuffled-label diagnostic.

**Table 5. Shuffled-label sanity check.**

| Setting | Test Accuracy | Weighted F1 | Confusion Matrix |
|---|---:|---:|---|
| Concat with shuffled training labels | 0.6992 | 0.6516 | [[76, 212], [28, 482]] |

The performance drops substantially from 98.12% to 69.92%. The result is still above 50% because the test set is class-imbalanced and predictions collapse toward the majority class. This diagnostic suggests that the high window-trend result is not solely caused by file paths, image layout, or trivial split artifacts, but it also warns that accuracy alone is insufficient; class-level metrics and confusion matrices are necessary.

## 7. Interpretability Analysis

### 7.1 Expert Weights as View-level Explanations

U-MoE provides a view-level explanation for every prediction. For a given sample, the final probability can be decomposed into:

\[
p(y|x) = \sum_v \alpha_v p_v(y|x^v).
\]

Here, \(\alpha_v\) answers "which chart view did the model rely on?" and \(p_v\) answers "what did that view predict?" This is more transparent than concat fusion, where view information is mixed inside a single MLP.

On window-trend classification, local U-MoE produces balanced average weights:

\[
\alpha_{candle}=0.291,\quad
\alpha_{MA}=0.289,\quad
\alpha_{RSI}=0.261,\quad
\alpha_{MACD}=0.159.
\]

This pattern suggests that the model uses several complementary views rather than relying only on one indicator.

### 7.2 Entropy as Reliability Explanation

Expert entropy provides a reliability signal. If an expert produces high entropy, its prediction is uncertain and its gate score is discounted. This is especially useful in next-day prediction. Although accuracy is poor, the model's uncertainty is meaningful: all experts approach maximum entropy, indicating that the model does not find reliable view-specific evidence.

This gives U-MoE a distinctive interpretability role. It explains both confident success on visible trend recognition and uncertainty-driven failure on future prediction.

### 7.3 Grad-CAM Analysis

We use Grad-CAM to visualize discriminative regions for each image view. In successful window-trend cases, useful explanations should focus on candle bodies, trend slopes, moving-average structure, RSI turning regions, or MACD transitions rather than blank areas or image borders.

Grad-CAM is not treated as standalone proof of financial reasoning. Instead, it complements expert weights and entropy:

- Gate weights identify which view is important.
- Entropy indicates whether the view is confident.
- Grad-CAM shows where the view-specific evidence appears in the image.

Together, these explanations form a multi-level interpretation: view selection, view reliability, and spatial evidence.

### 7.4 Retrieval as Representation Evidence

Retrieval evaluates whether learned features organize visually similar trend structures. On window-trend classification, U-MoE and concat models obtain high Hit@1, around 0.96-0.97. On next-day prediction, Hit@1 is near 0.50. This contrast supports our central claim: image representations capture visible chart similarity, but similar chart shapes do not reliably imply the same next-day return direction.

## 8. Discussion

### 8.1 Accuracy vs Interpretability

The strongest raw classifier is concat fusion, not U-MoE. This is expected: feature concatenation preserves all view embeddings and gives the classifier maximum flexibility. However, concat fusion behaves as a black-box feature-level fusion method. U-MoE trades a small amount of window-trend accuracy for explicit view-level interpretability.

Therefore, the proposed method should not be positioned as a universal accuracy improvement over concat fusion. Its contribution is different: it exposes which technical image experts support the decision and whether those experts are confident.

### 8.2 Interpreting Failure Is Also Valuable

In financial machine learning, knowing when a model lacks reliable evidence is critical. The next-day experiments show that U-MoE does not improve predictive accuracy, but it reveals high expert uncertainty. This is a useful result for responsible financial AI. An interpretable model should not only explain confident decisions; it should also indicate when prediction is unreliable.

### 8.3 Implications for Candlestick Image Research

Our results suggest that future candlestick-image research should clearly separate:

1. Recognizing visible trend states inside a chart.
2. Predicting out-of-window future returns.
3. Explaining which chart views drive a prediction.

Without this separation, high classification accuracy can be misinterpreted as market predictability. U-MoE helps make this separation explicit by providing expert-level and uncertainty-level diagnostics.

## 9. Limitations

This study has several limitations. First, the dataset covers ten U.S. stocks and one historical period. Broader evaluation across markets, sectors, and regimes is needed. Second, we focus on next-day prediction as the strict future task. Longer horizons may align better with technical patterns and should be explored. Third, technical indicator rendering choices may affect model behavior. Fourth, entropy is a simple uncertainty measure; future work could investigate calibration, Bayesian approximations, conformal prediction, or abstention mechanisms. Finally, Grad-CAM provides qualitative evidence but does not establish causal market reasoning.

## 10. Conclusion

This paper proposes U-MoE, an uncertainty-aware multi-view expert fusion framework for interpretable candlestick image trend classification. By treating candlestick, MA, RSI, and MACD charts as view-specific experts, U-MoE produces not only a final prediction but also expert probabilities, expert entropies, and gate weights. These quantities make the prediction process more transparent than standard concat fusion.

Experiments show that candlestick images can support highly accurate visible trend recognition, but strict next-day prediction remains near random. U-MoE achieves strong window-trend performance and provides interpretable expert weights. More importantly, when applied to next-day prediction, U-MoE reveals uniformly high expert uncertainty, explaining why the model cannot make reliable image-only future predictions.

The main message is therefore cautious and interpretable: candlestick image models should not be judged solely by accuracy. A useful financial image model should also explain which visual views support its decision and when the evidence is too uncertain to trust. U-MoE offers a step toward such interpretable and uncertainty-aware financial chart analysis.

## References

References should be filled according to the target venue style. Recommended citation groups include:

1. Candlestick chart image classification and stock chart pattern recognition.
2. Image-based time-series trend classification.
3. Mixture-of-experts models and sparse/dynamic expert routing.
4. Multi-view representation learning.
5. Predictive entropy, uncertainty estimation, and model calibration.
6. Grad-CAM and visual explanation methods.
7. Financial machine learning evaluation, chronological splits, and leakage-aware backtesting.
