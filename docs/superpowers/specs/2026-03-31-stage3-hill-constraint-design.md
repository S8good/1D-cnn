# 阶段3：Hill 物理约束接入设计

日期：2026-03-31

## 背景

阶段 2 的同步 `Cycle` 联合训练未能稳定优于 `Model C`。阶段 2.5 引入交替更新后，形成了三个候选主干：

- `2.5A`：激进版，整体不通过
- `2.5B`：平衡版，接近放行线但严格按门槛仍未通过
- `2.5C`：保守版，精度不提升，但最稳定、最适合作为阶段 3 母体配置

阶段 2.5 的结论已经明确：

1. `Cycle` 当前更适合作为“稳定化机制”，而不是单独证明精度提升的模块。
2. 若推进阶段 3，应以 `2.5C` 作为接入 `L_hill` 的起点。
3. 阶段 3 不应预设总体精度一定提升，但允许以更高风险探索是否存在更强上限。

当前仓库中尚无 `L_hill`、可微峰位提取或 Hill 参数学习模块，阶段 3 需要新增完整链路。

## 目标

阶段 3 的目标分为两层：

### 主目标

在 `2.5C` 母体配置上接入可微 `L_hill`，验证：

1. 物理约束链路可训练、可复现、可解释
2. `Hill-consistency error` 可改善
3. 极端浓度段表现有机会改善

### 次目标

在固定版 Hill 约束可用后，逐步放开 Predictor 和 Hill 参数，探索是否存在总体回归指标上的更高上限。

## 非目标

本阶段设计不做以下事情：

1. 不重新设计阶段 2.5 的交替训练机制
2. 不引入新的判别器、GAN 或额外生成器结构
3. 不同时把峰高、峰宽等多个物理量一起纳入阶段 3 主损失
4. 不做大规模网格超参数搜索

## 方案比较

### 方案 1：`2.5C-frozen` 直接接可学习 Hill

优点：

- 实现成本最低
- 最快开始阶段 3

缺点：

- Predictor 完全冻结时，总体回归精度上限很低
- 很难判断总体指标变化来自 `L_hill` 还是来自参数学习失败

### 方案 2：直接 `regressor + 可学习 Hill`

优点：

- 理论上限最高
- 最快进入“冲指标”模式

缺点：

- 同时引入多个不确定因素
- 失败原因最难拆解
- 不利于后续论文叙事和回退

### 方案 3：分层推进

推荐采用该方案。

推进顺序：

1. `3A-fixed-frozen`
2. `3B-fixed-regressor`
3. `3C-learnable-regressor`

优点：

1. 先验证 `L_hill` 链路是否成立
2. 再验证 Predictor 放开后是否带来总体收益
3. 最后才把 Hill 参数改成全局可学习，保留清晰回退路径

## 推荐设计

## 1. 总体架构

阶段 3 继续使用现有训练入口：

- `scripts/train_joint_physics_dl.py`

不新增新的训练主脚本，而是在现有联合训练框架上新增 Hill 分支。新增的核心文件为：

1. `src/core/stage3_hill.py`
2. `src/core/stage3_config.py`
3. `scripts/run_stage3_experiment.py`
4. 阶段 3 对比图与汇总脚本

职责划分如下：

- `stage3_hill.py`
  - 定义 Hill 函数
  - 定义固定参数 / 可学习参数模块
  - 定义可微峰位提取
  - 定义 `L_hill` 与物理一致性指标
- `stage3_config.py`
  - 定义阶段 3 profile
  - 定义固定参数与可学习模式默认值
  - 定义进入 `3B/3C` 的超参默认配置
- `train_joint_physics_dl.py`
  - 新增 `L_hill`
  - 新增阶段 3 CLI 参数
  - 新增阶段 3 输出 tag
- `run_stage3_experiment.py`
  - 封装阶段 3 实验命令
  - 读取源仓库数据与模型工件
  - 保存 seed 级证据快照

## 2. 版本定义

### 3A-fixed-frozen

- 母体：`2.5C`
- Predictor：`frozen`
- Hill 参数：固定
- 目标：
  - 验证 `L_hill` 链路是否可训练
  - 验证可微峰位提取是否稳定
  - 验证物理一致性图是否可解释

### 3B-fixed-regressor

- 母体：`3A`
- Predictor：`regressor`
- Hill 参数：固定
- 目标：
  - 在保持物理一致性的前提下，尝试把约束转化为总体回归收益

### 3C-learnable-regressor

- 母体：`3B`
- Predictor：`regressor`
- Hill 参数：全局可学习
- 目标：
  - 继续冲更高上限
  - 观察是否存在优于固定版的总体指标与物理一致性平衡点

## 3. `L_hill` 数据流

阶段 3 中，Generator 支路新增以下计算链：

1. 输入 `y_log`
2. `Generator(y_log)` 输出 `gen_raw`
3. 从 `gen_raw` 中提取 `lambda_ag_hat`
4. 从物理输入中取 `lambda_bsa`
5. 计算 `delta_lambda_hat = lambda_ag_hat - lambda_bsa`
6. 用 Hill 函数生成 `delta_lambda_target`
7. 计算 `L_hill = MSE(delta_lambda_hat, delta_lambda_target)`

因此阶段 3 首版总损失为：

`L_total = 1.0*L_conc + w_hill*L_hill + w_cycle*L_cycle + w_mono*L_mono + w_recon*L_recon`

### 设计原则

阶段 3 先只约束 `delta_lambda`，不把峰强和峰宽同时塞进主物理损失。原因：

1. 峰位是最核心且最稳定的物理量
2. 可微实现最直接
3. 更容易区分“物理约束有用”与“损失设计太复杂”

## 4. 可微峰位提取

阶段 3 不允许用硬 `argmax`。推荐方案为“窗口化 soft-argmax”。

### 流程

1. 先限定 Ag 峰搜索窗口
2. 取窗口内的预测谱强度
3. 做温度缩放 softmax
4. 以波长为权重求加权均值，得到 `lambda_ag_hat`

### 设计约束

1. 必须在物理合理窗口内提峰，避免峰位漂移到非目标区域
2. 必须保持梯度可回传
3. 必须对边界情况稳定，不产生 NaN

### 首版不做的事

1. 不对整条谱做全局质心
2. 不引入复杂的局部多峰分解
3. 不把温度参数纳入第一轮搜索

## 5. Hill 参数模式

阶段 3 采用标准 Hill 形式：

`Δλ(c) = Δλ_max * c^n / (K^n + c^n)`

其中：

- `Δλ_max`：最大位移上限
- `K`：半饱和浓度
- `n`：Hill 系数

### 3A / 3B：固定参数

固定参数从离线拟合或已有物理先验得到。

固定版的作用是：

1. 验证 `L_hill` 本身是否有价值
2. 给可学习版提供稳定初始化

### 3C：可学习参数

可学习版不一次性把全部参数放开，而采用分层解锁：

1. 先放开 `K` 和 `n`
2. 若有正向收益，再放开 `Δλ_max`

### 安全约束

可学习参数采用正值约束：

- `K = softplus(k_raw) + eps`
- `n = 1 + softplus(n_raw)`
- `Δλ_max = softplus(d_raw)`

同时加入弱正则：

- `L_hill_param_reg = ||theta - theta_init||^2`

目的是：

1. 防止参数跑到物理不合理区域
2. 保留向固定版回退的解释路径

## 6. 训练与搜索计划

阶段 3 搜索采用“小而有判别力”的策略，不做大规模网格。

### 第一步：`3A-fixed-frozen`

执行方式：

1. 先跑 `seed=20260325`
2. 只验证链路与物理一致性
3. 若训练稳定，再扩到 `3-seed`

重点观察：

1. 峰位是否稳定
2. `L_hill` 是否正常下降
3. `Hill-consistency error` 是否改善

### 第二步：`3B-fixed-regressor`

只在 `3A` 稳定后进行。

小范围搜索项：

1. `w_hill`
2. `predictor_lr`
3. `joint_epochs`

流程：

1. 先单 seed 预筛
2. 有正向信号后再做 `3-seed`

### 第三步：`3C-learnable-regressor`

只在 `3B` 已经出现正向信号后进行。

小范围搜索项：

1. `w_hill`
2. Hill 参数正则强度
3. 是否放开 `Δλ_max`

流程：

1. 单 seed 预筛
2. 若优于 `3B`，再扩到 `3-seed`

## 7. 指标与停止条件

### 优先指标

1. `MAE`
2. `RMSE`
3. `R2`
4. 低浓度段误差
5. 高浓度段误差
6. `Hill-consistency error`
7. 峰位稳定性

### 停止条件

1. 如果 `3A` 已明显破坏总体指标且物理一致性也无改善，则阶段 3 先停
2. 如果 `3B` 单 seed 无任何正向信号，则不扩到 `3-seed`
3. 如果 `3C` 单 seed 不优于 `3B`，则不扩到 `3-seed`
4. 一旦出现“总体不明显退化 + 物理一致性明显改善”的版本，就可停止并汇报
5. 若进一步出现“总体指标也优于固定版”的版本，则将其定为阶段 3 当前最优

## 8. 测试与验证

阶段 3 必须先写测试再实现。

### 单元测试

1. Hill 函数数值测试
2. 可学习参数约束测试
3. soft-argmax 峰位提取测试
4. 峰位梯度回传测试

### 集成测试

1. `train_joint_physics_dl.py` 可接受阶段 3 参数
2. 阶段 3 模式下会调用 `L_hill`
3. stage3 输出 tag 正确
4. stage3 runner 能保存 seed 快照

### 实验验证

1. `3A-fixed-frozen` 单 seed
2. `3A-fixed-frozen` 三 seed
3. `3B-fixed-regressor` 单 seed 搜索
4. `3B-fixed-regressor` 三 seed 复验
5. 必要时再做 `3C`

## 9. 产物

阶段 3 预计新增或更新：

1. `src/core/stage3_hill.py`
2. `src/core/stage3_config.py`
3. `scripts/run_stage3_experiment.py`
4. `docx/STAGE3_EXPERIMENT_SUMMARY_CN.md`
5. `outputs/compare_modelc_stage25_stage3.png`
6. `outputs/stage3_<variant>_seed<seed>/...`

## 10. 用户已确认的关键决策

1. 阶段 3 允许接受一定试错，以换取总体精度提升的机会
2. 不直接走“全局可学习参数直上”
3. 采用“先固定、后全局”的两阶段到三阶段推进
4. 阶段 3 的 Predictor 策略采用：
   - `3A` 冻结
   - `3B/3C` 放开 `regressor`
5. 阶段 3 先只约束 `delta_lambda`

## 11. 作用边界

本设计文档定义的是阶段 3 的训练与搜索策略，不直接规定：

1. 最终固定参数的具体数值
2. 最终最佳 `w_hill`
3. 最终是否一定执行到 `3C`

这些内容将在实现与实验阶段根据证据决定。
