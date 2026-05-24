# 多视图 K 线模型过拟合诊断与防护结果

## 诊断目标

当前最优多视图模型为 `candle / MA / RSI / MACD` 四视图独立编码后 concat 融合，窗口趋势识别测试准确率为 98.12%。该结果较高，因此需要判断模型是否存在过拟合、标签泄漏或伪特征学习。

本轮补充了三类诊断：

1. 正则化训练：增强、label smoothing、更高 dropout、更高 weight decay。
2. 冻结编码器线性探针：不允许 ResNet 编码器继续更新，只训练分类头。
3. 训练标签打乱 sanity check：打乱训练集标签，检查模型是否还能靠文件路径、日期或版式获得高分。

## 实验结果

| 实验 | 设置 | Test Acc | Weighted F1 | Retrieval Hit@1 | 说明 |
| --- | --- | ---: | ---: | ---: | --- |
| 最优 concat 模型 | 多视图预训练 + concat 融合 | 0.9812 | 0.9813 | 0.9674 | 最高性能模型，但训练后期有过拟合迹象 |
| 正则化 concat 模型 | 数据增强 + label smoothing 0.05 + dropout 0.45 + weight decay 5e-4 | 0.9712 | 0.9712 | 0.9599 | 性能略降，但仍稳定较高，可作为抗过拟合主报告模型 |
| 冻结编码器线性探针 | 冻结 ResNet，仅训练分类头 | 0.9398 | 0.9400 | 0.9273 | 表征本身有效，不完全依赖端到端记忆 |
| 训练标签打乱 | train 标签随机打乱，val/test 保持真实标签 | 0.6992 | 0.6516 | 0.9261 | 分类性能大幅下降，说明高分不是来自路径/日期/版式泄漏 |

## Train-Val-Test Gap

### 最优模型

最佳模型在第 2 轮达到最优验证损失，测试准确率 98.12%。后续训练集准确率继续快速升高，说明如果无限训练会出现过拟合。因此论文中应报告 early stopping 的 best checkpoint，而不是最后 epoch。

### 正则化模型

正则化模型测试准确率为 97.12%。虽然训练后期训练准确率仍可达到很高水平，但加入数据增强、label smoothing 和更强 dropout 后，测试性能仍保持稳定，说明模型不是只依赖微小像素记忆。

按股票测试准确率：

| Ticker | Test Acc |
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

所有股票均保持 93% 以上准确率，说明结果不是由某一只股票或少数样本主导。

### 冻结编码器线性探针

冻结编码器后，训练准确率约 93%，测试准确率约 94%。这说明多视图一致性预训练得到的图像表征本身已经包含趋势结构信息；如果模型主要靠端到端记忆训练集，冻结编码器后性能应明显崩溃。

## 标签打乱 Sanity Check

将训练集标签随机打乱后，模型验证和测试表现大幅下降：

- 测试准确率从 97%-98% 降至 69.92%。
- Weighted F1 从 97%-98% 降至 65.16%。
- Confusion matrix 显示模型明显偏向预测多数类，而非学习到稳定分类边界。

这个实验说明模型高分不太可能来自文件名、日期顺序、路径结构或固定版式泄漏。如果存在严重泄漏，训练标签打乱后模型仍可能在验证/测试上保持高分；实际结果没有发生这种情况。

## 可解释性如何体现没有明显过拟合

Grad-CAM 应作为定性证据，而不是单独证明。建议论文中展示正则化模型的 Grad-CAM，而不是最高分模型。展示原则：

1. 每个样本展示四个视图：candle、MA、RSI、MACD。
2. 热力图应主要覆盖价格趋势斜率、均线拐点、RSI 高低区间、MACD 柱体转折或信号线附近。
3. 避免只展示正确样本，应加入错误样本或低置信样本，说明模型何时失效。
4. 不应出现热力图长期集中在图片边框、空白背景、固定角落等伪特征区域。

本轮已经导出正则化模型 Grad-CAM 样例：

- `remote_results/pizon2025_multiview_windowtrend/overfit_checks/regularized/gradcam_AAPL_20230524_004485.png`
- `remote_results/pizon2025_multiview_windowtrend/overfit_checks/regularized/gradcam_AMD_20230524_004483.png`
- `remote_results/pizon2025_multiview_windowtrend/overfit_checks/regularized/gradcam_AMZN_20230524_004484.png`
- `remote_results/pizon2025_multiview_windowtrend/overfit_checks/regularized/gradcam_BAC_20230524_004491.png`

## 论文中建议采用的表述

可以写：

> Although the unregularized concat-fusion model achieves the highest test accuracy, its rapid convergence suggests potential overfitting risk. Therefore, we further evaluate a regularized variant with data augmentation, label smoothing, stronger dropout, and larger weight decay. The regularized model still achieves 97.12% test accuracy and maintains stable performance across all ten stocks. Moreover, a frozen-encoder linear probe achieves 93.98% test accuracy, indicating that the learned multi-view representation itself captures trend-related visual structures. Finally, when training labels are randomly shuffled, test accuracy drops substantially and predictions collapse toward the majority class, suggesting that the reported performance is unlikely to be caused by file-name, split, or layout leakage.

中文可以写：

> 尽管未正则化的 concat 融合模型取得最高测试准确率，但其快速收敛和训练集高准确率提示存在过拟合风险。因此，本文进一步引入数据增强、标签平滑、更高 dropout 和更强权重衰减进行正则化验证。正则化模型仍取得 97.12% 的测试准确率，并在 10 只股票上保持稳定表现。此外，冻结编码器的线性探针仍达到 93.98% 的测试准确率，说明多视图预训练表征本身已经包含趋势相关视觉信息。训练标签打乱后，分类性能显著下降并趋向多数类预测，表明模型高分不太可能来自文件名、切分方式或图像版式泄漏。

## 最终建议

论文主表可以同时报告两个模型：

1. **Best-performance model**：concat 融合，98.12%。
2. **Robust/regularized model**：增强 + label smoothing + dropout + weight decay，97.12%。

正文中强调：最终结论不依赖单一最高分，而由正则化、冻结探针、标签打乱、分股票稳定性和 Grad-CAM 共同支撑。

