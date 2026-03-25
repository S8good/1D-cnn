# LSPR_Spectra_Master 原理详解

本文档介绍 `DeepLearning/LSPR_Spectra_Master` 的核心原理、代码结构、数据流、训练与推理机制，以及当前版本（含 v2 + 校准 + 强度对齐）的实际行为。

## 1. 项目目标

项目目标是围绕 LSPR 光谱做一个“数字孪生”系统，支持两类能力：

1. 正向：给定浓度，生成对应的光谱（`concentration -> spectrum`）。
2. 逆向：给定测得光谱，反推浓度（`spectrum -> concentration`）。

在 GUI 中，这两类能力被组合成：

1. 滑块模拟（物理模型 + AI生成器）
2. 导入光谱后反演浓度
3. 导入光谱后预测光谱（链式：先反演浓度，再用生成器生成）

---

## 2. 目录与关键文件

- `run_gui.py`：PyQt 主界面，包含物理近似模型与 AI 功能入口。
- `src/core/ai_engine.py`：统一推理引擎（模型加载、浓度预测、光谱生成、spectrum-to-spectrum）。
- `src/core/full_spectrum_models.py`：神经网络结构定义：
  - `SpectralPredictor`（v1）
  - `SpectrumGenerator`
  - `SpectralPredictorV2`（新增稳健版本）
- `src/core/dataset.py`：全光谱训练数据集读取与归一化。
- `scripts/train_full_spectrum_ai.py`：训练 v1 预测器 + 生成器。
- `scripts/train_concentration_v2.py`：训练 v2 浓度回归器（新增）。
- `scripts/fit_predictor_v2_calibration.py`：拟合 v2 的后处理单调校准层（新增）。
- `scripts/evaluate_test_predict.py`：统一评估脚本（MAE/MAPE/R2/Bland-Altman 等）。

---

## 3. 数据与标签约定

核心训练数据来自：

- `data/processed/All_Absorbance_Spectra_Preprocessed.xlsx`

要求：

1. 第一列为 `Wavelength`。
2. 光谱列名包含浓度信息，格式可被正则提取，例如：`0.5ng/ml-Ag-01_01`。
3. `Ag` 列用于训练 AI 全光谱模型（逆向与生成器）。

浓度标签在深度学习内部一般使用：

\[
\log_{10}(c + 10^{-3})
\]

这样可减轻浓度跨数量级时的回归难度。

---

## 4. 模型体系（核心）

### 4.1 v1 浓度预测器 `SpectralPredictor`

- 输入：单通道归一化光谱 `[B,1,L]`
- 主体：1D CNN + 池化 + 全连接
- 输出：`log10(conc + 1e-3)`

特点：

1. 结构简单，训练快。
2. 对噪声和基线漂移敏感。
3. 依赖“单条光谱 min-max 归一化”，容易丢失绝对强度信息。

### 4.2 光谱生成器 `SpectrumGenerator`

- 输入：浓度（log域，1维）
- 主体：全连接展开 + 多级 Upsample + Conv1d
- 输出：归一化光谱，再反归一化回真实量级

训练损失（`scripts/train_full_spectrum_ai.py`）：

1. 重建损失 `MSE(gen, real)`
2. 质心约束（center-of-mass）损失，抑制峰位偏移失控

### 4.3 v2 浓度预测器 `SpectralPredictorV2`（新增）

- 输入改为双通道 `[B,2,L]`：
  1. 原始强度通道（robust 标准化）
  2. 一阶导数通道（robust 标准化）
- 主体：更深 1D CNN + GELU + Dropout
- 训练增强：加入单调惩罚（monotonic penalty）

目的：

1. 保留绝对强度信息（解决 v1 对幅值不敏感）。
2. 加强对谱形变化趋势的利用（导数通道）。
3. 让浓度预测随真实浓度更单调、更稳定。

### 4.4 v2 后处理校准层（新增）

- 类型：Isotonic Regression（单调回归）
- 作用：对 v2 输出的 log 浓度做单调映射校正
- 文件：`models/predictor_v2_calibration.pth`

作用机理：

1. 网络先给一个原始预测 `log_c_raw`。
2. 用校准函数 `f(.)` 得到 `log_c_cal = f(log_c_raw)`。
3. 再反变换到浓度域。

这一步可显著降低系统性偏差，特别是高浓度段。

---

## 5. 推理流程（`ai_engine.py`）

### 5.1 统一入口

`FullSpectrumAIEngine` 在初始化时尝试加载：

1. v1 predictor + generator + norm params（基础能力）
2. v2 predictor + v2 norm params（可选增强）
3. v2 calibration（可选后处理）

优先级：

1. 能用 v2 就用 v2
2. v2 出错自动回退 v1

### 5.2 逆向浓度预测 `predict_concentration`

- v2 路径：
  1. 光谱重采样到模型波长长度
  2. 构建 raw + gradient 双通道
  3. robust 标准化
  4. v2 网络预测 `log_c`
  5. 若有校准层则校准
  6. 反变换得到浓度

### 5.3 光谱生成 `generate_spectrum`

1. 输入浓度转 log
2. 送入生成器
3. 用 `spec_min/spec_max` 反归一化
4. 得到预测光谱

### 5.4 光谱到光谱 `predict_spectrum_from_spectrum`

注意这是**链式方法**，不是端到端映射：

1. 输入光谱 -> 反演浓度
2. 预测浓度 -> 生成器出光谱
3. 对生成光谱做强度对齐（新增）：
   - `pred_spectrum_raw`：原始生成器输出
   - `pred_spectrum`：对齐后的输出

---

## 6. GUI 行为解释（`run_gui.py`）

### 6.1 启动后的三条曲线

默认启动会执行 `update_plot(5.0)`，显示：

1. 灰线：BSA 基线（Lorentz 重建）
2. 红虚线：物理公式的 post 光谱（同样是 Lorentz）
3. 紫线：AI 生成器光谱

当前版本中，紫线在显示前会做一次“分位数强度对齐”（新增），减少峰高量级不一致。

### 6.2 你之前看到峰高差距很大的原因

历史版本里：

1. 首屏紫线是生成器原始输出
2. 没和物理曲线做幅值尺度对齐

所以会出现“峰形像，但峰高量级差很大”的现象。

### 6.3 当前强度对齐方法

显示层使用稳健线性标定：

\[
y_{aligned} = s \cdot y_{pred} + b
\]

其中：

1. `s` 用 P90-P10 动态范围比值得到
2. `b` 用 P50（中位强度）对齐得到

这不是“只对齐峰值”，而是对整个强度分布做对齐。

---

## 7. 训练脚本职责划分

### 7.1 `scripts/train_full_spectrum_ai.py`

训练并保存：

1. `spectral_predictor.pth`（v1）
2. `spectral_generator.pth`
3. `norm_params.pth`

### 7.2 `scripts/train_concentration_v2.py`（新增）

训练并保存：

1. `spectral_predictor_v2.pth`
2. `predictor_v2_norm_params.pth`

### 7.3 `scripts/fit_predictor_v2_calibration.py`（新增）

拟合并保存：

1. `predictor_v2_calibration.pth`

---

## 8. 评估体系（`scripts/evaluate_test_predict.py`）

测试集目录：`data/test-predict`

输出：

1. `outputs/eval_test_predict/detail_metrics.csv`
2. `outputs/eval_test_predict/segment_metrics.csv`
3. `outputs/eval_test_predict/summary_report.txt`
4. `outputs/eval_test_predict/bland_altman.png`
5. `outputs/eval_test_predict/true_vs_pred.png`

可评估指标：

1. MAE / RMSE / MAPE / R2
2. 分段误差（低中高浓度）
3. 光谱 MAE / RMSE / 相关性

---

## 9. 当前版本的关键改进（相对原始版）

1. 浓度反演：v1 -> v2（双通道 + robust 标准化 + 单调惩罚）
2. 后处理：加入 isotonic 校准层
3. 光谱形态显示：加入强度对齐，显著减少峰高失配
4. 评估体系：新增标准化误差报表与图

---

## 10. 已知局限与建议

### 10.1 已知局限

1. `spectrum -> spectrum` 仍是链式，不是端到端。
2. 校准层目前基于现有数据拟合，需外部数据验证泛化。
3. 部分中间浓度样本（如 10 ng/ml 附近）仍可能有个例偏差。

### 10.2 建议路线

1. 加入独立验证集与批次外测试（不同天、不同设备）。
2. 尝试端到端 spec2spec 模型并与链式方案 AB 对比。
3. 在 GUI 增加“原始/对齐后光谱切换”开关，便于诊断。
4. 将模型加载的 `torch.load` 兼容策略进一步规范（安全与兼容平衡）。

---

## 11. 常用命令

在项目根目录执行：

```powershell
# 训练 v1（预测器 + 生成器）
& 'C:/ProgramData/anaconda3/envs/gan/python.exe' scripts/train_full_spectrum_ai.py

# 训练 v2 浓度模型
& 'C:/ProgramData/anaconda3/envs/gan/python.exe' scripts/train_concentration_v2.py

# 拟合 v2 校准层
& 'C:/ProgramData/anaconda3/envs/gan/python.exe' scripts/fit_predictor_v2_calibration.py

# 评估 test-predict
& 'C:/ProgramData/anaconda3/envs/gan/python.exe' scripts/evaluate_test_predict.py

# 启动 GUI
& 'C:/ProgramData/anaconda3/envs/gan/python.exe' run_gui.py
```

---

## 12. 一句话总结

这个项目本质是“物理启发 + 深度学习”的混合数字孪生系统：

- 物理模块负责可解释的峰位/峰强变化模拟；
- 深度学习模块负责全光谱反演与生成；
- 当前版本通过 v2 回归器、单调校准和强度对齐，把浓度预测稳定性和光谱峰高一致性都显著提升。
