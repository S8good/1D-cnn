# LSPR_Spectra_Master 原理详解（按当前代码实现）

本文档基于当前仓库代码（`DeepLearning/LSPR_Spectra_Master`）给出一份可直接用于研发对齐的架构说明，覆盖系统目标、模块边界、训练与推理流程、模型工件、评估口径和已知限制。

## 1. 项目目标与业务定义

项目目标是建立 CEA 等生物标志物浓度与 LSPR 吸收光谱之间的双向映射能力，并在 GUI 中提供“物理可解释 + AI 可泛化”的联动展示。

当前系统支持三类业务场景：

1. 正向模拟（`Concentration -> Spectrum`）  
输入目标浓度，输出一条预期反应后光谱。该输出可来自物理残差重建结果，也可来自 AI 生成器。
2. 逆向反演（`Spectrum -> Concentration`）  
输入实测光谱，输出浓度估计值（含线性定量区间与超量程分档解释）。
3. 链式映射（`Spectrum -> Concentration -> Spectrum`）  
输入实测光谱，先反演浓度，再基于该浓度生成对应光谱，并与输入谱做同轴对比。

说明：

1. 本项目的 `Spectrum -> Spectrum` 不是端到端模型，而是链式组合。
2. 当前代码不存在 GAN 判别器训练，光谱生成器是监督回归生成器。

## 2. 系统分层架构

系统可以拆成 4 层。

1. 表现层（GUI）  
`main.py` 启动 `src/gui/main_window.py`，负责交互、绘图、文件导入、消息提示。
2. 应用服务层  
`src/core/digital_twin_service.py` 负责编排物理引擎与 AI 引擎，向 GUI 返回统一结构化结果。
3. 核心引擎层  
`src/core/reconstruction.py` 提供物理残差引擎；`src/core/ai_engine.py` 提供统一 AI 推理引擎。
4. 模型与算法层  
`src/core/full_spectrum_models.py` 与 `src/core/neural_network.py` 定义神经网络；`src/core/physics_models.py` 定义 Lorentz/特征提取/强度对齐等函数。

## 3. 关键模块职责

### 3.1 GUI 层

入口是 `main.py`，不是 `run_gui.py`。  
`LSPRDigitalTwinMainWindow` 提供三个主交互：

1. 浓度滑块驱动实时模拟曲线刷新。
2. 导入光谱执行浓度反演。
3. 导入光谱执行链式 spec2spec 预测。

默认绘图叠加 2 到 3 条曲线：

1. BSA 基线谱（物理重建）。
2. 反应后物理谱（物理重建）。
3. AI 生成谱（可进行强度对齐后显示）。

### 3.2 应用服务层（`DigitalTwinService`）

`DigitalTwinService` 的作用是屏蔽底层复杂性，对 GUI 提供稳定接口：

1. `build_plot_context(concentration)`  
同时调用物理残差引擎和 AI 生成器，返回 `PlotContext`。
2. `infer_concentration_from_file(file_path)`  
读取文件并调用 AI 引擎执行逆向反演。
3. `predict_spectrum_from_file(file_path)`  
读取文件并调用 AI 引擎执行链式 spec2spec。

### 3.3 物理残差引擎（`ResidualPhysicsEngine`）

该引擎以“可解释特征变化”为中心，流程如下：

1. 从训练源提取 pre/post 光谱特征（峰位、峰强、FWHM）。
2. 构造输入特征 `x=[log10(c+1e-3), lambda_pre, A_pre, fwhm_pre]`。
3. 构造监督目标 `y=[delta_lambda, delta_A]`。
4. 用 `StandardScaler` 对 `x/y` 标准化。
5. 用 `LSPRResidualNet` 拟合残差映射。
6. 预测时把 delta 叠加到基线特征，再调用 Lorentz 函数重建整条谱。

输出是 `ResidualPrediction`，含：

1. `delta_lambda`
2. `delta_amplitude`
3. `peak_wavelength`
4. `peak_intensity`
5. `fwhm`

### 3.4 统一 AI 引擎（`FullSpectrumAIEngine`）

AI 引擎负责模型装载、输入适配、回退策略与业务解释输出。

模型加载策略：

1. 基础必需工件：`spectral_predictor.pth`、`spectral_generator.pth`、`norm_params.pth`。
2. 可选增强工件：`spectral_predictor_v2.pth`、`predictor_v2_norm_params.pth`。
3. 可选后处理工件：`predictor_v2_calibration.pth`。

推理优先级：

1. 优先走 v2。
2. v2 失败时自动回退 v1。
3. 若校准层存在，则在 v2 输出后执行单调校准。

业务解释策略：

1. 线性定量上限 `ULOQ=18.0 ng/ml`。
2. `<=18` 返回定量值。
3. `>18` 返回超量程分档（`18-40`、`40-75`、`>75`）。

## 4. 深度学习模型定义

### 4.1 v1 预测器 `SpectralPredictor`

1D CNN 回归器，输入单通道光谱，输出 `log10(conc+1e-3)`。  
适合基础场景，结构简单。

### 4.2 光谱生成器 `SpectrumGenerator`

输入 1 维浓度（log 域），经全连接升维后通过多级 `Upsample + Conv1d` 逐步还原成 1D 光谱。  
输出经过 `Sigmoid`，再按 `spec_min/spec_max` 反归一化到真实强度范围。

### 4.3 v2 预测器 `SpectralPredictorV2`

双通道输入架构：

1. 原始强度通道。
2. 一阶导数通道。

训练端和推理端都使用 robust 统计量：

1. `median`
2. `IQR`

激活使用 `GELU`，损失中额外加入单调惩罚项，约束浓度预测随真实浓度整体单调上升。

## 5. 训练流程与工件

### 5.1 全光谱训练（v1 predictor + generator）

脚本：`scripts/train_full_spectrum_ai.py`

训练内容：

1. `SpectralPredictor` 用回归损失训练浓度反演。
2. `SpectrumGenerator` 用重建损失训练光谱生成。
3. 生成器额外使用质心损失（COM）抑制峰位漂移失控。

主要产物：

1. `models/pretrained/spectral_predictor.pth`
2. `models/pretrained/spectral_generator.pth`
3. `models/pretrained/norm_params.pth`
4. `models/checkpoints/full_spectrum_epoch_*.pth`

### 5.2 v2 浓度模型训练

脚本：`scripts/train_concentration_v2.py`

关键机制：

1. 由训练集估计 `raw_med/raw_iqr/diff_med/diff_iqr`。
2. 构建双通道输入并训练 `SpectralPredictorV2`。
3. 主损失为 `HuberLoss`，叠加 `monotonic_penalty`。

主要产物：

1. `models/pretrained/spectral_predictor_v2.pth`
2. `models/pretrained/predictor_v2_norm_params.pth`
3. `models/checkpoints/predictor_v2_epoch_*.pth`
4. `models/checkpoints/predictor_v2_best.pth`

### 5.3 v2 校准层拟合

脚本：`scripts/fit_predictor_v2_calibration.py`

流程：

1. 读取 v2 模型与 norm 参数。
2. 在数据上得到 `pred_log` 与真实 `log_conc` 的锚点关系。
3. 用 `IsotonicRegression` 拟合单调映射。
4. 保存 `x_thresholds/y_thresholds`。

主要产物：

1. `models/pretrained/predictor_v2_calibration.pth`
2. `models/checkpoints/predictor_v2_calibration_latest.pth`

## 6. 端到端推理数据流

### 6.1 逆向反演（Spectrum -> Concentration）

1. 读入光谱并转为 1D 数组。
2. 若长度不一致则按目标波长轴线性重采样。
3. 进入 v2 或 v1 分支进行 log 浓度预测。
4. 做反对数变换得到 `ng/ml`。
5. 应用 ULOQ 规则生成报告文本。

### 6.2 正向生成（Concentration -> Spectrum）

1. `c` 映射为 `log10(c+1e-3)`。
2. 送入生成器得到归一化谱。
3. 使用 `spec_min/spec_max` 反归一化。

### 6.3 链式 spec2spec（Spectrum -> Concentration -> Spectrum）

1. 输入谱先反演浓度。
2. 再按该浓度生成预测谱。
3. 最后对预测谱执行稳健强度对齐：
   使用输入谱与预测谱的 `P10/P50/P90` 估计线性 `scale + offset`。
4. 返回原始预测谱、对齐后谱、浓度、报告模式等完整字段。

## 7. 数据组织与切分策略

数据列名采用可解析格式，例如 `0.5ng/ml-Ag-01_07`。  
`split_reconstructed_dataset.py` 按浓度分层（`stratify`）进行 pair 级切分，输出：

1. `train_preprocessed_pairs.*`
2. `val_preprocessed_pairs.*`
3. `test_preprocessed_pairs.*`
4. `split_assignment.csv`

该方式保证 train/val/test 在浓度分布上可比，降低偶然分布偏差。

## 8. 评估脚本与指标口径

脚本：`scripts/evaluate_test_predict.py`

默认读取 `data/test-predict`，输出：

1. 逐样本明细（浓度误差与光谱误差）。
2. 分浓度段统计表。
3. 文本总结报告。
4. Bland-Altman 图。
5. True-vs-Pred 散点图。

主要指标：

1. MAE
2. RMSE
3. MAPE
4. R2
5. 光谱 MAE/RMSE/相关系数

## 9. 当前版本工程特性

1. 双路线并行  
物理残差路线和深度学习路线同时可用，便于解释与对照。
2. 推理弹性  
v2 可选增强并带自动回退，减少线上因单模型故障导致的不可用。
3. 结果表达分层  
不仅输出数值，还输出“定量/超量程”解释，贴近检验场景。
4. 强度后对齐  
spec2spec 与 GUI 叠图时显著减少峰高量级错配。

## 10. 已知边界与后续建议

当前边界：

1. spec2spec 是链式，不是直接学习输入谱到输出谱的条件映射。
2. v2 校准层当前由现有数据拟合，跨批次泛化能力需独立验证。
3. 全光谱训练脚本与 v2 脚本采用了不同标准化范式，维护时需注意一致性。

建议优先级：

1. 建立外部批次验证集并固定版本化评估基线。
2. 在 GUI 增加“原始预测谱/对齐谱”切换。
3. 增加端到端 spec2spec 基线模型用于 A/B 对比。
4. 整理数据预处理协议，统一各训练脚本归一化口径。

## 11. 常用命令

```powershell
# 训练 v1 predictor + generator
python scripts/train_full_spectrum_ai.py

# 训练 v2 predictor
python scripts/train_concentration_v2.py

# 拟合 v2 isotonic 校准层
python scripts/fit_predictor_v2_calibration.py

# 评估 test-predict
python scripts/evaluate_test_predict.py

# 启动 GUI
python main.py
```

## 12. 总结

本项目当前架构是“残差物理重建 + 深度学习反演/生成 + 工程化推理回退”的混合数字孪生方案。  
其中 v2 双通道预测、isotonic 校准和强度对齐是稳定性提升的三项核心工程增量。
