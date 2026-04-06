# LSPR 论文与工程联合落地计划（阶段版）

本文档用于指导 `LSPR_Spectra_Master` 从当前版本推进到可发表高水平 SCI/EI 交叉学科论文的完整实施路径。目标是把“特征融合注入（Fusion）+ PGDL（Cycle + Hill）”做成可复现、可解释、可对比的工程与论文一体化方案，并在 Hill 接入前显式加入 `阶段2.5` 交替联合训练闸门。

## 1. 总体目标（终局）

在保持系统可用性的前提下，实现以下成果：

1. 建立并验证 `Model D (Ours)`：`V2_Fusion + PGDL(Hill + Cycle + Monotonic)`。
2. 用完整消融实验证明性能提升链路：`A < B < C < D`，并补充 `C+Cycle` 与 `C+Hill`。
3. 形成论文所需核心图表：架构图、消融表、Bland-Altman、True-vs-Pred、物理一致性图、单调违例率表。
4. 交付可复现实验脚本、固定数据切分、固定随机种子、可追溯模型工件。

### 当前进度摘要（截至 2026-04-02）

1. 阶段0、阶段1已完成，`Model C (V2_Fusion)` 相比 `Model B (V2)` 在 MAE/RMSE/R2 三项核心指标上已有明确提升。
2. 阶段2已完成 `3-seed` 复核，结论是 `C+Cycle(regressor)` 未带来稳定收益，因此主线不再围绕同步更新版 `Cycle` 继续小步调参。
3. 阶段2.5/阶段3的工程基础设施已落地：已实现 Hill 核心模块、Stage 3 配置、交替训练原语、训练入口集成、固定 Hill 参数拟合脚本与实验 runner。
4. 已完成修正后训练链路下的 `3A/3B/3C` 公平 `3-seed` 对比；关键修正是训练期 `lambda_BSA` 必须使用原始 nm，而不是归一化特征。
5. `3-seed` 结果表明：`3C-learnable-regressor` 当前在 `RMSE`、`R2` 与 `Hill-consistency error` 三项指标上最优，是阶段3的主线候选配置。
6. 当前代码验证状态：已在 `py39` 环境中完成自动化验证，执行 `conda run -n py39 pytest` 得到 `32 passed, 1 warning`；`gan` 环境可正常导入 `torch`，可作为备选运行环境。当前不建议使用 base/3.12 环境作为正式验证环境。
7. **阶段4核心结果产物已基本生成**：已输出 `ablation_summary.csv`、`ablation_comparison_figure.png`、`segment_stats_table.csv`、`segment_stats_figure.png`、`true_vs_pred_3c_figure.png`、`bland_altman_3c_figure.png`、`hill_consistency_figure.png`，已能支撑结果章节主叙事。
8. **阶段5论文打包已进入收口阶段**：`scripts/plot_stage3_comparison.py`、`scripts/plot_ablation_full.py`、`scripts/plot_supplementary.py` 已产出主要图表，`docx/METHODS_RESULTS_EN.md`、`docx/RESULTS_DRAFT_CN.md`、`docx/FORMULAS_CN_PLAIN.md` 已形成论文文字与公式草稿；当前主要剩余工作是统一图文口径并补齐少量附加实验/复现说明。

## 2. 阶段拆解总览

1. 阶段0：基线冻结与实验规范化（1-2天）
2. 阶段1：特征融合改造（Fusion）（3-5天）
3. 阶段2：联合训练框架（同步更新版 Cycle 基线）（4-6天）
4. 阶段2.5：交替联合训练主干（Hill 前置闸门）（3-5天）
5. 阶段3：Hill 物理约束接入（固定参数 -> 可学习参数）（4-6天）
6. 阶段4：系统化消融实验（5-7天）
7. 阶段5：可视化与论文打包（3-5天）

建议总周期：5-7周（根据算力与调参轮次浮动）。

## 3. 各阶段详细计划

## 阶段0：基线冻结与实验规范化

目标：建立“可信对照组”，避免后续结果不可比较。

任务：

1. 固定数据切分文件与随机种子：沿用 `splits_reconstructed`，统一训练/验证/测试。
2. 统一评价脚本出口：MAE、RMSE、MAPE、R2、单调违例率（新增）。
3. 固定基线模型版本并打标签：
   - Model A：`SpectralPredictor(V1)`
   - Model B：`SpectralPredictorV2`
4. 明确实验记录模板：实验编号、超参数、数据版本、结果文件路径。

模型定义说明（阶段0固定，不得在消融中变更）：

1. Model A（`SpectralPredictor(V1)`）：
   - 模型类：`src/core/full_spectrum_models.py::SpectralPredictor`
   - 输入：单通道 Ag 光谱（`shape=[N,1,L]`）
   - 结构：3 层 1D-CNN + MLP 回归头
   - 训练入口（阶段0固定 split 版本）：`scripts/train_concentration_v1.py`
   - 预训练工件：`models/pretrained/spectral_predictor_v1_split.pth`
2. Model B（`SpectralPredictorV2`）：
   - 模型类：`src/core/full_spectrum_models.py::SpectralPredictorV2`
   - 输入：双通道特征（原始强度 + 一阶导数，`shape=[N,2,L]`）
   - 结构：更深 1D-CNN（GELU）+ MLP 回归头
   - 训练入口：`scripts/train_concentration_v2.py`（基于固定 split）
   - 预训练工件：`models/pretrained/spectral_predictor_v2.pth`

预期目标：

1. 同一脚本重复运行，关键指标波动在可接受范围内（建议 MAE 波动 < 5%）。
2. A/B 模型结果可稳定复现。

交付物：

1. 基线结果表（CSV）
2. 实验配置说明（Markdown）
3. 固定随机种子训练命令

验收标准：

1. 团队任一成员可在同环境复现实验并得到同趋势结论。

阶段0已完成结果（截至 2026-03-25）：

1. 数据版本与切分：`data/processed/splits_reconstructed`（`within_cv10=True` 子集，固定 train/val/test = 181/39/39 对）。
2. 固定随机种子：`seed=20260325`。
3. Test 指标：
   - Model A (`SpectralPredictor(V1)`)：MAE `11.2791`，RMSE `22.2850`，MAPE `40.9267%`，R2 `0.3342`
   - Model B (`SpectralPredictorV2`)：MAE `6.4774`，RMSE `13.2968`，MAPE `30.1632%`，R2 `0.7630`
4. 阶段结论：`B` 显著优于 `A`，阶段0“可信对照组”目标达成。
5. 对应产物：
   - 指标汇总：`outputs/baseline_stage0_ab_metrics.csv`
   - 明细：`outputs/split_test_metrics_predictor_v1.csv`、`outputs/split_test_metrics_predictor_v2.csv`
   - 对比图：`outputs/compare_ab_test_metrics.png`

---

## 阶段1：特征融合改造（Fusion）

目标：先做“结构创新”并验证其独立价值（对应 Model C）。

任务：

1. 改造数据集：
   - 在 `Dataset` 中输出 `physics_bsa = [lambda_BSA, A_BSA, FWHM_BSA]`。
   - `__getitem__` 从 `(spectrum_ag, conc)` 变为 `(spectrum_ag, physics_bsa, conc)`。
2. 新建/替换模型：
   - 实现 `SpectralPredictorV2_Fusion`。
   - CNN 分支提取谱特征；物理分支编码 BSA 特征；在全连接前 `concat`。
3. 训练仅浓度任务：
   - 不加 Generator、不加 Hill、不加 Cycle。
   - 保留 monotonic penalty（可选，建议保留但权重小）。

预期目标：

1. `Model C` 在 MAE/RMSE/R2 上优于 `Model B`。
2. 在批次漂移样本或噪声增强样本上，C 的鲁棒性提升更明显。

交付物：

1. `SpectralPredictorV2_Fusion` 模型定义
2. 对应训练脚本和权重
3. `B vs C` 对比指标与图

验收标准：

1. 至少 2 个核心指标优于 B（如 MAE 降低 + R2 提升）。

阶段1已完成结果（截至 2026-03-25）：

1. 已实现 `Model C`：`src/core/full_spectrum_models.py::SpectralPredictorV2_Fusion`。
2. 已完成数据改造：训练脚本读取 `Reconstructed_Preprocessed_Features_and_Delta.xlsx(sheet1)`，为每个 Ag 谱注入 `physics_bsa=[lambda_BSA, A_BSA, FWHM_BSA]`。
3. 训练入口：`scripts/train_concentration_v2_fusion.py`（当前为浓度任务，未接 Cycle/Hill）。
4. Test 指标（同一 split、同一 seed）：
   - Model B (`SpectralPredictorV2`)：MAE `6.4774`，RMSE `13.2968`，MAPE `30.1632%`，R2 `0.7630`
   - Model C (`SpectralPredictorV2_Fusion`)：MAE `5.5855`，RMSE `10.8219`，MAPE `28.3212%`，R2 `0.8430`
5. 阶段结论：`C` 在 MAE/RMSE/R2 三项核心指标均优于 `B`，阶段1验收标准已满足。
6. 对应产物：
   - 指标汇总：`outputs/baseline_stage0_stage1_abc_metrics.csv`
   - 明细：`outputs/split_test_metrics_predictor_v2_fusion.csv`
   - 对比图：`outputs/compare_abc_test_metrics.png`、`outputs/compare_bc_test_metrics.png`

---

## 阶段2：联合训练框架（Cycle 先行）

目标：搭建 PGDL 训练主干，但先接入 `L_cycle`，控制训练稳定性。

任务：

1. 新建脚本 `scripts/train_joint_physics_dl.py`。
2. 加载预训练初始化：
   - `V2_Fusion`（Predictor）
   - `SpectrumGenerator`（Generator）
3. 定义联合损失（不含 Hill）：
   - `L_conc`：真实谱 -> Predictor -> 浓度监督
   - `L_cycle`：输入浓度 -> Generator 生成假谱 -> Predictor 回归浓度
   - `L_mono`：单调惩罚
4. 总损失（阶段2建议）：
   - `L_total = 1.0*L_conc + 0.1*L_cycle + 0.05*L_mono`
5. 同步更新 Predictor + Generator，观察收敛曲线与梯度稳定性。

预期目标：

1. 训练稳定，无明显梯度爆炸/塌陷。
2. 在不引入 Hill 的情况下，较 C 有小幅增益或至少不退化。

交付物：

1. `train_joint_physics_dl.py`（可跑）
2. 联合训练日志与收敛图
3. `C vs C+Cycle` 指标表

验收标准：

1. 成功训练并保存可用联合模型权重。

阶段2最新结论更新（截至 2026-03-26）：

1. 已完成 `3-seed` 成对实验（`seed=20260325, 20260331, 20260407`），统一配置为：
   - `C+Cycle(regressor)`：`predictor_train_mode=regressor`，`w_cycle=0.005`，`generator_lr=5e-5`，`joint_epochs=60`。
2. 结果汇总（mean ± std）：
   - `Model C`：MAE `6.4808 ± 0.8631`，RMSE `13.2536 ± 2.3615`，MAPE `33.6853 ± 4.6550`，R2 `0.7595 ± 0.0833`
   - `C+Cycle(regressor)`：MAE `6.5297 ± 0.9626`，RMSE `13.6224 ± 2.8297`，MAPE `33.7320 ± 5.0088`，R2 `0.7441 ± 0.1036`
3. 阶段结论（论文/报告口径）：
   - 引入 `Cycle` 后未带来稳定收益，当前阶段应先收口，不再继续围绕同一损失权重做小步调参。
4. 下一步分流建议：
   - 若下一步目标是论文或报告：采用上述结论，按“未获得稳定增益”如实汇报。
   - 若下一步目标是继续攻关提升：应转向训练机制改造，不再磨现有 loss 权重；优先实现交替更新的 `P-step/G-step` 联合训练。
5. 对应产物：
   - `3-seed` 明细：`outputs/stage2_regressor_3seed_20260326_143033/paired_seed_metrics.csv`
   - 汇总表：`outputs/stage2_regressor_3seed_20260326_143033/summary_mean_std_with_mape.csv`
   - 独立对比图：`outputs/stage2_regressor_3seed_20260326_143033/compare_c_vs_c_cycle_3seed_summary.png`

---

## 阶段2.5：交替联合训练主干（Hill 前置闸门）

目标：在接入 `L_hill` 之前，先把联合训练从“同步更新”改造成“交替更新”，把 Stage 3 建立在一个更稳定、可解释、可回退的训练主干上。

任务：

1. 将联合训练从单次共享反向传播改造成 `P-step / G-step` 交替更新。
2. 显式控制 Predictor 可训练范围与更新频次，例如 `frozen / regressor / tail / all`。
3. 预留 `p_steps / g_steps / hill_mode / hill_reg_weight` 等 Stage 3 需要的配置位。
4. 形成保守可用的 `2.5C` 母体配置，作为 Stage 3 `3A/3B/3C` profile 的基础。

预期目标：

1. 联合训练主干稳定，不再因 `Cycle` 分支直接扰动 Predictor 主任务而出现不可控退化。
2. 为后续 Stage 3 的 `fixed Hill` 与 `learnable Hill` 提供统一入口和可复现实验脚本。

交付物：

1. 交替训练入口与 profile 约束逻辑
2. 可复现的 Stage 2.5/Stage 3 参数模板
3. 可直接被 Stage 3 调用的训练原语

验收标准：

1. 至少能稳定支撑 `3A-fixed-frozen` 的 smoke run 闭环。
2. 作为母体配置时，不引入新的训练崩溃点或不可恢复依赖。

阶段2.5 / Stage 3 联动实现状态（截至 2026-03-31）：

1. 已在代码中落地交替训练原语：`src/core/stage3_training.py` 提供 hill-aware generator step 与 alternating epoch runner。
2. 已定义 `3A-fixed-frozen`、`3B-fixed-regressor`、`3C-learnable-regressor` 三个 Stage 3 profile，均以 `2.5C` 为母体配置。
3. 当前 `3A-fixed-frozen` smoke run 已成功打通，说明 `2.5C` 风格主干已满足“可承载 Stage 3”的最小工程门槛。
4. 在此主干上完成的 Stage 3 `3-seed` 对比已经证明：阶段2.5 不只是“可承载 Stage 3”，而且确实为后续 Hill 约束提供了可稳定优化的母体配置。

---

## 阶段3：Hill 物理约束接入

目标：补全 PGDL 的物理一致性约束，形成 `Model D`。

任务：

1. 可微峰位提取实现：
   - 避免硬 `argmax`，使用 `soft-argmax` 或局部质心近似。
2. 定义 `L_hill`：
   - 从 Generator 输出谱提取 `lambda_Ag`。
   - 计算 `Delta_lambda = lambda_Ag - lambda_BSA`。
   - 与 Hill 理论曲线值做 MSE。
3. Hill 参数策略（二选一）：
   - 固定离线拟合参数（优先，稳定）
   - 或全局可学习参数（需边界正则）
4. 总损失（阶段3建议）：
   - `L_total = 1.0*L_conc + 0.1*L_hill + 0.1*L_cycle + 0.05*L_mono`
5. 重点监控高低浓度端误差与单调违例率。

预期目标：

1. `Model D` 在极端浓度段优于 `Model C/C+Cycle`。
2. 物理一致性图中预测点更贴近 Hill 曲线。

交付物：

1. `L_hill` 代码实现
2. `Model D` 最优权重
3. 物理一致性对比图（B/C/D）

验收标准：

1. 模型总体性能不下降，且物理一致性指标显著改善。

阶段3实施进展更新（截至 2026-03-31）：

1. 已实现 Hill 核心模块：`src/core/stage3_hill.py`
   - 包含 `hill_delta_lambda`、`build_delta_lambda_table`、`soft_argmax_peak_nm`
   - 包含 `FixedHillCurve` 与 `LearnableHillCurve`
2. 已实现 Stage 3 profile 与固定参数拟合脚本：
   - `src/core/stage3_config.py`
   - `scripts/fit_stage3_hill_params.py`
3. 已实现 Stage 3 训练集成与 runner：
   - `src/core/stage3_training.py`
   - `scripts/train_joint_physics_dl.py`
   - `scripts/run_stage3_experiment.py`
4. 已生成固定 Hill 参数工件：`models/pretrained/stage3_hill_params.pth`
5. 已完成 `3A-fixed-frozen`、`3B-fixed-regressor`、`3C-learnable-regressor` 的公平 `3-seed` 对比，统一 seed 为：
   - `20260325`
   - `20260331`
   - `20260407`
6. `3-seed` 汇总结果：
   - `3A-fixed-frozen`：MAE `6.5492 ± 0.0000`，RMSE `13.4009 ± 0.0000`，R2 `0.7592 ± 0.0000`，Hill-MAE `2.2556 ± 0.0778`
   - `3B-fixed-regressor`：MAE `6.4325 ± 0.0383`，RMSE `13.3490 ± 0.1977`，R2 `0.7611 ± 0.0071`，Hill-MAE `1.8591 ± 0.2502`
   - `3C-learnable-regressor`：MAE `6.5292 ± 0.0675`，RMSE `13.2483 ± 0.2347`，R2 `0.7646 ± 0.0083`，Hill-MAE `1.6982 ± 0.0262`
7. 关键工程发现：
   - `3C` 早期效果不佳的主要原因不是 learnable Hill 参数本身，而是训练期 `Hill loss` 误用了归一化后的 `lambda_BSA` 特征。
   - 在修正 `lambda_BSA` 坐标后，并让 `3C` 从 `3A` generator 权重启动且进行短 warmup，`Hill-MAE` 从 `7.8~8.2 nm` 显著降到 `1.6982 nm`。
8. 当前结论：
   - 若优先看物理一致性，`3C` 是当前最佳配置。
   - 若综合看 `RMSE + R2 + Hill-MAE`，`3C` 也是当前阶段3最强主线。
   - 后续阶段4应以 `C vs 3A vs 3B vs 3C` 为核心对比面板，并继续补图表与分段分析。

---

## 阶段4：系统化消融实验（论文核心）

目标：产出可直接进论文结果章节的核心证据。

任务：

1. 跑全套模型：
   - Model A：V1
   - Model B：V2
   - Model C：V2_Fusion
   - Model C+Cycle
   - Model C+Hill
   - Model D：V2_Fusion + Hill + Cycle + Mono
2. 每个模型固定同数据切分、同评估脚本、同随机种子策略。
3. 输出总表：MAE、RMSE、MAPE、R2、单调违例率。
4. 分段统计：低浓度、中浓度、高浓度。
5. 加噪鲁棒性测试：如高斯噪声、基线漂移、强度缩放。

预期目标：

1. 明确展示每个模块的增益来源。
2. 形成稳定排序：`A < B < C <= C+Cycle/C+Hill < D`（允许中间两者接近）。

交付物：

1. `ablation_summary.csv`
2. 分段结果表
3. 鲁棒性结果图

验收标准：

1. 审稿视角下可回答“为什么有效、哪部分有效、在什么场景有效”。

阶段4进展（截至 2026-04-02）：

1. 已完成主结果面板与表格输出：
   - `outputs/ablation_summary.csv`
   - `outputs/ablation_comparison_figure.png`
   - `outputs/true_vs_pred_3c_figure.png`
   - `outputs/bland_altman_3c_figure.png`
   - `outputs/hill_consistency_figure.png`
2. 已完成分段统计输出：
   - `outputs/segment_stats_table.csv`
   - `outputs/segment_stats_figure.png`
3. 当前阶段4可以支撑的核心结论：
   - `A -> B`：双通道输入带来显著收益；
   - `B -> C`：BSA 物理分支进一步提升 MAE/RMSE/R2；
   - `3A/3B/3C`：在 Stage 3 内部对比中，`3C-learnable-regressor` 在 `RMSE`、`R2`、`Hill-MAE` 上综合最优。
4. 当前阶段4尚未完全收口的部分：
   - `C+Hill` 作为独立消融项尚未单独固化为最终面板；
   - 鲁棒性/噪声扰动实验尚未形成正式结果图；
   - `Monotonicity Violation Rate (MVR)` 尚未形成最终论文表格。
5. 阶段判断：
   - 阶段4已从“计划态”进入“结果已基本成型、尚需补完少量边角项”的状态。

---

## 阶段5：可视化与论文打包

目标：把工程结果转化为高质量论文叙事。

任务：

1. 方法图：
   - 双通道谱分支 + BSA 特征融合分支
   - Predictor-Generator 循环闭环
   - Hill 物理约束注入点
2. 结果图：
   - Bland-Altman
   - True-vs-Pred
   - Hill 物理一致性图
   - 消融柱状图/表格
3. 临床解释图表：
   - 单调违例率
   - 高浓度超量程行为解释
4. 论文材料：
   - 方法伪代码
   - 训练细节（超参、硬件、重复次数）
   - 复现声明（代码路径与命令）

预期目标：

1. 结果可支撑投稿到交叉学科传感/分析方向期刊。
2. 图表和方法描述形成一致闭环，不留“黑箱解释断点”。

交付物：

1. 论文图表全集
2. 结果章节草稿
3. 可复现实验附录

验收标准：

1. 内部预审可通过"可复现性 + 方法创新性 + 物理一致性"三项核查。

阶段5进展（截至 2026-04-02）：

已完成主要图表与文稿草稿，包含以下内容：

1. Stage 3 主图与表格（脚本：`scripts/plot_stage3_comparison.py`）：
   - `outputs/stage3_comparison_figure.png` — Figure 1：1x5 分组柱状图（3A/3B/3C，误差棒，Hill-MAE 灰底，最优值标星）
   - `outputs/stage3_hilmae_figure.png` — Figure 1b：Hill-MAE 独立放大图（可独立插图引用）
   - `outputs/stage3_seed_detail_figure.png` — Figure 2：3x5 Seed 明细散点图（含均值线与范围带）
   - `outputs/stage3_paper_table.csv` — Table 1：论文摘要表（mean±std 格式，`*` 标最优值）
2. 阶段4主结果与补充图（脚本：`scripts/plot_ablation_full.py`、`scripts/plot_supplementary.py`）：
   - `outputs/ablation_comparison_figure.png`
   - `outputs/true_vs_pred_3c_figure.png`
   - `outputs/bland_altman_3c_figure.png`
   - `outputs/hill_consistency_figure.png`
   - `outputs/segment_stats_figure.png`
3. 文稿草稿已形成：
   - `docx/METHODS_RESULTS_EN.md`
   - `docx/RESULTS_DRAFT_CN.md`
   - `docx/FORMULAS_CN_PLAIN.md`
4. 核心结论（可直接写入论文）：
   - 3C 在 RMSE（13.2483±0.2347）、R2（0.7646±0.0083）、Hill-MAE（1.6982±0.0262 nm）三项最优且 Hill-MAE 方差最小；
   - 综合判定 **3C-learnable-regressor 为阶段3最强主线**。
5. 当前阶段5剩余工作：
   - 统一方法图、Methods 文本与代码实现的口径，避免示意图与实现细节不一致；
   - 补齐复现说明，包括推荐环境、执行命令、依赖清单；
   - 决定论文主模型命名口径（`Model D` 与 `3C-learnable-regressor` 的映射表述）并全稿统一；
   - 视投稿目标补充少量审稿人敏感项，如 MVR 表或鲁棒性附图。

---

## 4. 关键指标定义（建议统一）

主指标：

1. MAE (ng/ml)
2. RMSE (ng/ml)
3. MAPE (%)
4. R2

物理与可靠性指标：

1. Monotonicity Violation Rate (MVR)
2. Hill-consistency error（`Delta_lambda` 对 Hill 曲线残差）
3. 极端浓度段误差（低端/高端）

工程稳定性指标：

1. 训练收敛成功率
2. 多次重复实验方差

## 5. 风险与回退策略

风险1：联合训练不稳定。  
回退：先冻结 Generator 只训 Predictor；再逐步解冻。

风险2：Hill 约束导致总性能下降。  
回退：降低 `w_hill`，先保证 `L_conc` 主任务收敛。

风险3：Fusion 提升不显著。  
回退：增强 BSA 分支容量，或加入 feature normalization/gating。

风险4：消融结果不呈现理想排序。  
回退：补充分段与噪声场景，突出 D 在极端条件下优势。

## 6. 推荐周节奏（可执行版）

第1周：阶段0 + 阶段1（拿下 Model C）  
第2周：阶段2（完成 `C+Cycle` 结论收口）  
第3周：阶段2.5（交替训练主干）+ 阶段3固定 Hill smoke run  
第4周：阶段3多 seed 实验 + 阶段4主消融  
第5周：阶段4补图表 + 阶段5论文包装

## 7. 里程碑判定

M1：Model C 明确优于 B。✅  
M2：阶段2得出明确结论，确认同步更新版 `Cycle` 不再继续深挖。✅  
M3：阶段2.5主干可稳定承载 Stage 3，并产出固定 Hill smoke run。✅  
M4：Model D 在至少两个关键维度优于 C，或在物理一致性与极端浓度段形成明确收益。🟡（3C 物理一致性领先已确认；完整 A/B/C/D 消融对比仍待补全）  
M5：消融与图表可直接进入论文正文。🟡（Stage 3 内部图表已完成；完整消融全集仍待补全）

---

一句话执行策略：先做“结构创新可证据化”（Fusion），再用阶段2.5把联合训练主干稳定下来，随后接入 Hill 物理约束，最后用系统化消融把故事闭环。
