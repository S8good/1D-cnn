# 阶段 2.5 / 阶段 3 实验总结

更新日期：2026-03-31

## 1. 目的

阶段 2.5 的目标不是直接证明 `Cycle` 一定提升总体回归精度，而是回答两个问题：

1. 交替更新是否能修复阶段 2 中同步联合训练的干扰问题。
2. 是否存在一个足够稳定的联合训练主干，可作为阶段 3 接入 `L_hill` 的母体配置。

阶段 3 的目标是在这个主干之上引入 Hill 物理约束，并判断：

1. 物理一致性是否真正改善。
2. 回归主任务是否保持在可接受范围内。

---

## 2. 阶段 2.5 结论回顾

阶段 2.5 采用漏斗式推进：

- `2.5A`：激进版
- `2.5B`：平衡版
- `2.5C`：保守版

统一使用的 seed 为：

- `20260325`
- `20260331`
- `20260407`

阶段 2.5 的正式结论保持不变：

1. `2.5A` 没有形成稳定收益。
2. `2.5B` 是当时最接近可接受放行的配置，但没有在全部门槛上完全胜出。
3. `2.5C` 虽然没有带来精度增益，但提供了最干净、最稳定、最适合承载阶段 3 的联合训练主干。

因此，阶段 3 的母体配置继续采用 `2.5C`。

---

## 3. 阶段 3 关键工程修正

在阶段 3 的早期单 seed 尝试中，`3B` 与 `3C` 的 `Hill-MAE` 一度维持在 `7.8 ~ 8.2 nm`，明显不合理。后续定位到两个关键问题：

1. 训练期 `Hill loss` 错误使用了归一化后的 `lambda_BSA` 特征，而不是原始 `nm` 坐标。
2. `3C` 的 learnable Hill 参数虽然可训练，但如果没有合理初始化和 warmup，generator 容易沿错误方向更新。

修正措施为：

1. 在 Stage 3 训练原语中显式传递原始 `lambda_BSA_nm`。
2. `3C` 从 `3A` 的 generator 权重启动。
3. 在正式联训前增加短的 generator-only hill warmup。

这一步是阶段 3 结果转折的核心原因。

---

## 4. 阶段 3 三个 profile 的公平 3-seed 结果

统一 seed：

- `20260325`
- `20260331`
- `20260407`

统一对比对象：

- `3A-fixed-frozen`
- `3B-fixed-regressor`
- `3C-learnable-regressor`

结果汇总如下：

### 4.1 3A-fixed-frozen

- `MAE = 6.5492 ± 0.0000`
- `RMSE = 13.4009 ± 0.0000`
- `MAPE = 36.0685 ± 0.0000`
- `R2 = 0.7592 ± 0.0000`
- `MVR = 0.4737 ± 0.0000`
- `Hill-MAE = 2.2556 ± 0.0778`

### 4.2 3B-fixed-regressor

- `MAE = 6.4325 ± 0.0383`
- `RMSE = 13.3490 ± 0.1977`
- `MAPE = 35.3425 ± 0.0590`
- `R2 = 0.7611 ± 0.0071`
- `MVR = 0.4825 ± 0.0124`
- `Hill-MAE = 1.8591 ± 0.2502`

### 4.3 3C-learnable-regressor

- `MAE = 6.5292 ± 0.0675`
- `RMSE = 13.2483 ± 0.2347`
- `MAPE = 35.2916 ± 0.2147`
- `R2 = 0.7646 ± 0.0083`
- `MVR = 0.4825 ± 0.0124`
- `Hill-MAE = 1.6982 ± 0.0262`

---

## 5. 当前判断

从 `3-seed` 结果看：

1. `3C` 在 `Hill-MAE` 上最佳，且方差最小。
2. `3C` 在 `RMSE` 和 `R2` 上也优于 `3A` 和 `3B`。
3. `3B` 在 `MAE` 上略优于 `3C`，但物理一致性不如 `3C`。

因此，若当前阶段的优先级是“物理一致性优先，同时总体回归性能不明显恶化”，则 `3C-learnable-regressor` 是当前最强主线配置。

---

## 6. 结论写法建议

当前阶段建议在论文或内部汇报中采用如下表述：

1. 阶段 2.5 证明了交替联合训练主干是必要的，它为阶段 3 提供了稳定的母体配置。
2. 阶段 3 的关键收益并不来自单纯“放开 learnable Hill 参数”，而是来自：
   - 修正训练期的 `lambda_BSA` 坐标使用错误
   - 让 `3C` 从方向正确的 `3A` generator 启动
   - 在正式联训前执行短 warmup
3. 在修正后的公平 `3-seed` 对比中，`3C` 同时取得了最好的 `Hill-MAE`、`RMSE` 和 `R2`，因此可以作为阶段 4 消融与论文结果图表的主线配置。

---

## 7. 产物

本轮可直接引用的文件包括：

- `outputs/stage3_3seed_detail_20260331.csv`
- `outputs/stage3_3seed_summary_20260331.csv`
- `outputs/stage3_single_seed_summary_latest.csv`

以及各个 seed 的快照目录：

- `outputs/stage3_3a_fixed_frozen_seed20260325`
- `outputs/stage3_3a_fixed_frozen_seed20260331`
- `outputs/stage3_3a_fixed_frozen_seed20260407`
- `outputs/stage3_3b_fixed_regressor_seed20260325`
- `outputs/stage3_3b_fixed_regressor_seed20260331`
- `outputs/stage3_3b_fixed_regressor_seed20260407`
- `outputs/stage3_3c_learnable_regressor_seed20260325`
- `outputs/stage3_3c_learnable_regressor_seed20260331`
- `outputs/stage3_3c_learnable_regressor_seed20260407`

这些目录和表格足以支撑阶段 2.5 -> 阶段 3 的过渡结论，以及后续阶段 4 的正式消融统计。
