# 主文图序与整篇骨架（中文执行版）

> **项目**：LSPR_Spectra_Master 论文主文组织方案  
> **目标期刊**：Biosensors and Bioelectronics（B&B）  
> **文档用途**：锁定主文图序、Results 章节骨架、整篇章节顺序与全文叙事主线  
> **最后更新**：2026-04-02

---

## 1. 论文主定位

### 1.1 一句话定位

本文要讲清楚的不是“又一个浓度回归模型”，而是：

**在 LSPR 浓度预测任务中，将双通道光谱表示、BSA 基线物理特征和 Hill 物理一致性约束联合起来，可以在保持预测性能竞争力的同时，显著提升模型的物理一致性与可解释性。**

### 1.2 主线贡献

主文中建议始终围绕以下三点展开：

1. **表示层贡献**：双通道输入（强度 + 一阶导数）优于单通道原始谱。
2. **结构层贡献**：BSA 基线物理分支（Model C）进一步提升浓度预测性能。
3. **物理层贡献**：Hill 约束（Model D）虽然不一定在 MAE 上绝对最优，但在物理一致性上最强，形成“准确性 + 物理合理性”的平衡。

### 1.3 主文命名规则

全文必须统一命名，避免混用 `3C`、`Model D`、`learnable-regressor` 导致读者混乱。

建议采用以下口径：

1. **Model A**：Baseline 1D-CNN (V1)
2. **Model B**：Dual-channel 1D-CNN (V2)
3. **Model C**：V2-Fusion with BSA physics branch
4. **Model D**：Physics-guided model with learnable Hill constraint
5. **3A / 3B / 3C**：仅作为 **Model D 内部 Stage 3 配置名** 使用

推荐写法：

> “Model D was implemented using the 3C-learnable-regressor configuration.”

之后主文统一写 `Model D`，不要在正文反复切换到 `3C-learnable-regressor`。

---

## 2. 主文图序（最终推荐）

说明：以下是**主文推荐图序**，不完全等同于当前文件夹中的临时编号；目标是让叙事最顺。

### Figure 1. 总体框架图

- 文件：`outputs/paper_selected_figures/main_figures/Figure1_Framework_Main.png`
- 图题建议：
  `Overview of the proposed physics-guided LSPR prediction framework`
- 核心作用：
  - 一图讲清输入、双分支编码、融合回归、训练期 generator 与 Hill 物理一致性约束
  - 为后文 Model C / Model D 的差别做结构铺垫
- 应放位置：
  - Methods 3.4 Model Architecture

### Figure 2. 主消融图（A / B / C / D）

- 文件：`outputs/paper_selected_figures/main_figures/Figure2_Ablation_A_B_C_D.png`
- 图题建议：
  `Ablation study showing progressive gains from Model A to Model D`
- 核心作用：
  - 证明整体链路 `A -> B -> C -> D`
  - 其中 `A -> B` 说明双通道表示有效，`B -> C` 说明物理分支有效，`C -> D` 引出“物理一致性优先级”
- 应放位置：
  - Results 4.1

### Figure 3. 最佳主线模型的预测一致性图

- 文件：`outputs/paper_selected_figures/main_figures/Figure3_True_vs_Pred_ModelD.png`
- 图题建议：
  `True-versus-predicted concentration for the best physics-guided model`
- 核心作用：
  - 展示最佳主线模型在测试集上的浓度预测一致性
  - 用于支撑“具备实际预测能力，不只是物理一致性更好”
- 注意：
  - 当前图文件名是 `3c`，但正文图注应明确这是 **Model D / best physics-guided model**
- 应放位置：
  - Results 4.2

### Figure 4. Hill 物理一致性图

- 文件：`outputs/paper_selected_figures/main_figures/Figure4_Hill_Consistency.png`
- 图题建议：
  `Physical consistency of Stage 3 configurations against the Hill binding curve`
- 核心作用：
  - 这是论文“物理约束有效”的核心证据图
  - 必须明确突出 3C / Model D 在 Hill 曲线上的贴合度最好
- 应放位置：
  - Results 4.3

### Figure 5A. Stage 3 三配置总体比较图

- 文件：`outputs/paper_selected_figures/main_figures/Figure5A_Stage3_Comparison.png`
- 图题建议：
  `Comparison of Stage 3 physics-guided configurations`
- 核心作用：
  - 展示 3A / 3B / 3C 在 MAE、RMSE、MAPE、R²、Hill-MAE 上的系统差异
  - 为“为何最终选择 Model D = 3C”提供完整指标支持
- 应放位置：
  - Results 4.3 或 4.4

### Figure 5B. Hill-MAE 放大图

- 文件：`outputs/paper_selected_figures/main_figures/Figure5B_Hill_MAE_Zoom.png`
- 图题建议：
  `Zoomed comparison of Hill-MAE across Stage 3 configurations`
- 核心作用：
  - 作为 Figure 5 的强化子图
  - 如果主文版面紧，可并入 Figure 5 的子面板；若版面允许，可保留单独编号

---

## 3. 补充材料图序（建议）

### Figure S1

- 文件：`outputs/paper_selected_figures/supplementary_figures/FigureS1_Framework_Detailed.png`
- 用途：详细结构图

### Figure S2

- 文件：`outputs/paper_selected_figures/supplementary_figures/FigureS2_Bland_Altman.png`
- 用途：Bland-Altman 分析

### Figure S3

- 文件：`outputs/paper_selected_figures/supplementary_figures/FigureS3_Segmented_Error.png`
- 用途：分段误差统计

### Figure S4

- 文件：`outputs/paper_selected_figures/supplementary_figures/FigureS4_MVR_Comparison.png`
- 用途：单调违例率（MVR）比较

### Figure S5

- 文件：`outputs/paper_selected_figures/supplementary_figures/FigureS5_Extended_Ablation_C_Hill_D.png`
- 用途：`C / C+Cycle / C+Hill / D` 扩展消融

### 可选补图

- 文件：`outputs/paper_selected_figures/backup/Backup_Stage3_Seed_Detail.png`
- 用途：3A / 3B / 3C 三 seed 的逐指标散点分布
- 建议：只有在 reviewer 明确关注稳定性或 seed 离散度时再启用

---

## 4. 整篇章节顺序（主文）

推荐采用标准 IMRaD 结构，但结果部分使用“方法-结果耦合叙事”：

1. Abstract
2. Introduction
3. Materials and Methods
4. Results
5. Discussion
6. Conclusion

其中真正决定论文说服力的是：

1. **Methods 3.4**：把模型链路定义清楚
2. **Results 4.1–4.4**：把“为什么有效、哪里有效、为什么值得接受”讲清楚
3. **Discussion**：解释为什么 Model D 不是单纯追求最低 MAE，而是追求“物理一致性 + 准确性平衡”

---

## 5. 整篇主文骨架（推荐标题与每节内容）

## 1. Abstract

### 目标

用最短文字回答四个问题：

1. 为什么要做这个问题
2. 你提出了什么方法
3. 结果最好到什么程度
4. 意义是什么

### 建议内容顺序

1. LSPR 浓度预测受非线性饱和、批间漂移和物理一致性不足限制。
2. 提出双通道光谱编码 + BSA 物理分支 + Hill 物理一致性约束的框架。
3. 报告核心结果：
   - A→B→C 的准确性提升
   - D 在 Hill-MAE 上最优
4. 总结意义：
   - 模型不仅能预测，还能更符合 LSPR 结合物理规律

### 摘要里必须出现的数字

建议至少出现两组：

1. Model C 相对 Model B 的核心提升
2. Model D 在 Hill-MAE 上的最佳结果

---

## 2. Introduction

### 2.1 背景

要回答：

1. CEA 检测为什么重要
2. LSPR 传感为什么有价值
3. 仅靠传统回归或黑箱深度学习为什么不够

### 2.2 现有问题

重点列三点：

1. 光谱形状变化复杂，单通道输入不足
2. 批间漂移和基线状态变化导致泛化困难
3. 纯数据驱动模型缺乏物理一致性约束

### 2.3 本文解决方案

引出本文三层创新：

1. 双通道输入
2. BSA 物理特征融合
3. Hill 约束的物理引导训练

### 2.4 本文贡献（建议单独一段）

可以明确写成 3 条：

1. 提出基于 Ag 谱和 BSA 基线物理特征的融合预测框架
2. 构建具有 Hill 物理一致性约束的 physics-guided 训练机制
3. 通过系统消融和多 seed 实验验证准确性与物理一致性的平衡收益

---

## 3. Materials and Methods

### 3.1 Biosensor Assay, Dataset, and Data Split

应写内容：

1. 数据来源
2. 谱的测量阶段（BSA / Ag）
3. 数据筛选规则（CV < 10%）
4. 训练/验证/测试切分与固定 seed

### 3.2 Spectral Input Representation

应写内容：

1. 强度通道
2. 一阶导数通道
3. robust normalization
4. 为什么一阶导数对峰边缘和形状变化有帮助

### 3.3 Physics Feature Extraction from BSA Baseline

应写内容：

1. Lorentzian 拟合
2. 提取 `λ_BSA / A_BSA / FWHM_BSA`
3. 为什么这些特征可作为批次/基线状态描述符

### 3.4 Model Family and Architecture

建议分成：

1. 3.4.1 Model A：V1 baseline
2. 3.4.2 Model B：dual-channel V2
3. 3.4.3 Model C：fusion predictor
4. 3.4.4 Model D：physics-guided model with Hill constraint

### 3.5 Training Protocol

应写内容：

1. 各模型训练轮数、优化器、学习率
2. `3A / 3B / 3C` 作为 Model D 内部配置
3. predictor-step / generator-step

### 3.6 Evaluation Metrics

建议正文主列：

1. MAE
2. RMSE
3. MAPE
4. R²
5. Hill-MAE

补充材料列：

1. MVR
2. segmented MAE
3. Bland-Altman bias / LoA

---

## 4. Results（重点章节）

Results 不要写成“图一张接一张的流水账”，要按论证链路组织。

### 4.1 从 Model A 到 Model D 的逐级增益

**目标**：先建立主性能故事线。

**核心问题**：

1. 双通道表示是否有效？
2. BSA 物理分支是否有效？
3. 物理约束模型是否保持竞争力？

**使用图表**：

1. Figure 2：`ablation_comparison_figure.png`
2. Table 1：A/B/C/D 汇总表

**本节要写出的结论**：

1. A→B：双通道输入显著提升性能
2. B→C：BSA 物理分支进一步提高准确性
3. C→D：D 的准确性与 C 接近，但其真正优势不在纯 MAE，而在物理一致性

**本节最后一句建议**：

> “These results indicate that the proposed framework should not be judged solely by regression accuracy, but by the joint balance between predictive performance and physical consistency.”

### 4.2 最佳主线模型的预测一致性与临床可读性

**目标**：回答“模型预测到底准不准，读者能不能信”。

**使用图表**：

1. Figure 3：`true_vs_pred_3c_figure.png`
2. Supplementary Figure S2：`bland_altman_3c_figure.png`

**本节要写出的结论**：

1. 最佳 physics-guided 主线模型在主要浓度区间表现稳定
2. 高浓度区出现系统性偏差，主要来自 LSPR 饱和效应而非模型崩溃
3. Bland-Altman 结果说明偏差可接受且方向可解释

### 4.3 Hill 约束带来的物理一致性收益

**目标**：这是本文最关键的创新结果节。

**使用图表**：

1. Figure 4：`hill_consistency_figure.png`
2. Figure 5：`stage3_comparison_figure.png`
3. Figure 5b：`stage3_hilmae_figure.png`
4. Table S1：Stage 3 三配置对比

**本节要写出的结论**：

1. 3A → 3B → 3C 呈现越来越好的 Hill-MAE
2. 可学习 Hill 参数并非让模型更“复杂”而已，而是显著提高物理合理性
3. 3C / Model D 在 `RMSE + R² + Hill-MAE` 综合维度上最优，因此应被定义为最终主线模型

### 4.4 扩展消融：Cycle 与 Hill 的独立作用

**目标**：回应 reviewer 最可能问的问题：“Cycle 和 Hill 分别有用吗？”

**使用图表**：

1. Supplementary Figure S5：`c_hill_comparison_figure.png`
2. Supplementary Table S2：`c_hill_comparison_summary.csv`
3. Supplementary Table S3：`c_hill_3seed_summary.csv`

**本节要写出的结论**：

1. `C+Cycle` 未带来稳定收益
2. `C+Hill` 相比 `C+Cycle` 更接近最终主线
3. 最终 Model D 的优点来自“交替训练 + Hill 约束 + 合适的学习策略”，不是单一一个 loss 的魔法

### 4.5 补充结果：分段误差与单调性行为

**目标**：把临床相关性与行为约束补完整。

**使用图表**：

1. Supplementary Figure S3：`segment_stats_figure.png`
2. Supplementary Figure S4：`mvr_comparison_figure.png`
3. Supplementary Table S1：`mvr_summary.csv`

**本节要写出的结论**：

1. 高浓度区仍是主要误差来源
2. 单调性约束改善的是整体行为边界，而不是保证所有指标都绝对最优
3. 结果支持将“物理一致性 + 行为合理性”作为该类 LSPR 模型的重要评价维度

---

## 5. Discussion

Discussion 不要重复 Results，而要解释“为什么会这样”。

### 5.1 为什么 BSA 物理分支有效

应强调：

1. BSA 基线状态对 Ag 结合后的谱响应有先验解释意义
2. 该分支相当于把“批次差异/初始共振状态”显式输入模型

### 5.2 为什么 Model D 不是最低 MAE 但仍是最终主线

应强调：

1. 在物理引导任务中，最低 MAE 不是唯一目标
2. Hill 一致性决定模型在生物传感机制上的可信度
3. 因此 Model D 的价值在于“可解释的、物理一致的预测”

### 5.3 方法意义

建议上升到方法层面：

1. 本文展示了传统光谱解析与物理约束可与深度学习有效耦合
2. 该思路可推广到其他具有饱和响应关系的光学/生物传感任务

### 5.4 局限性

建议明确写，不要回避：

1. 数据规模有限
2. 高浓度区饱和效应仍限制精度上限
3. 当前验证集中在单体系、单实验条件

### 5.5 后续工作

1. 更大样本、多批次验证
2. 加入更真实的噪声/漂移鲁棒性实验
3. 探索更强的物理参数联合学习机制

---

## 6. Conclusion

结论建议压缩成三句话：

1. 双通道表示与 BSA 物理分支显著提升了 LSPR 浓度预测性能。
2. 引入 Hill 物理一致性约束后，Model D 在保持竞争性预测精度的同时，显著提升了物理合理性。
3. 该框架为构建可解释、可复现、物理一致的 LSPR 智能分析系统提供了可行路径。

---

## 7. 主文写作顺序（执行建议）

建议不要从 Introduction 开始写，而是按下面顺序推进：

1. **先锁图序和图注**
   - 先确认 Figure 1–5 及补充材料图
2. **先写 Results**
   - 因为结果决定全文叙事主线
3. **再反写 Methods**
   - 确保 Methods 与 Results 完全一致
4. **再写 Introduction**
   - 引言要服务已经确定的结果故事
5. **最后写 Abstract / Conclusion**
   - 这两部分最后写会更稳

---

## 8. 当前稿件最需要统一的三件事

1. **图号与图注统一**
   - 当前已有图很多，但主文和补充材料编号还需最终锁定

2. **命名统一**
   - 主文统一使用 `Model D`
   - `3A / 3B / 3C` 仅在内部配置对比中出现

3. **Methods 与实现对齐**
   - generator 结构、soft-argmax 位置、Hill loss 口径必须与最终框架图保持一致

---

## 9. 最终建议

如果只保留一个最核心的主文故事线，建议压缩成一句话：

> **双通道光谱表示解决“看得见”的问题，BSA 物理分支解决“看得准”的问题，Hill 约束解决“看得合理”的问题。**

这句话可以作为全文叙事总纲，贯穿摘要、引言、结果和讨论。
