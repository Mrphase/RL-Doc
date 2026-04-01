# RL Algorithm Selection for GPU DCVS Real-Time Adaptive Tuning

> **Author**: GitHub Copilot (Claude Opus 4.6) + zehuixie  
> **Date**: 2026-04-01  
> **Project**: QCOM-RL / RL-DCVS  
> **Purpose**: 供多 AI 综合决策参考

---

## 1. 问题背景

### 1.1 目标

在移动端（Qualcomm 8850 平台）游戏运行时，**实时自适应调整 GPU DCVS（Dynamic Clock and Voltage Scaling）性能参数**，在维持帧率目标（FPS ≥ 59）的前提下最小化功耗和频率，找到动态最优组合。

**明确排除静态优化**——不是找一组固定最优参数，而是根据实时 GPU 负载、温度、游戏场景变化动态切换。

### 1.2 控制参数（Action Space）

| 参数名 | sysfs 路径 | 取值范围 | 含义 |
|--------|-----------|---------|------|
| `first_step_down` | `/sys/class/kgsl/kgsl-3d0/gmu/dcvs_tunables/first_step_down` | [3, 5, 10, 15, 20, 25] (6 值) | GPU 降频首步幅度 |
| `penalty_down` | `.../penalty_down` | [85, 90, 95, 98] (4 值) | 降频惩罚阈值 |
| `penalty_up` | `.../penalty_up` | [85, 90, 95, 98] (4 值) | 升频惩罚阈值 |
| `strict_frame` | `.../strict_frame` | [0, 1] (2 值) | 严格帧率模式 |

**总离散动作数**：6 × 4 × 4 × 2 = **192 个离散 action**

### 1.3 状态空间（State Features）

来自 `node_dataset_sample` 采集的逐帧数据，共 70+ 列，核心特征：

| 特征类别 | 具体字段 | 维度 |
|---------|---------|------|
| GPU 频率 | `gpu_freq_hz`, `gpu_freq_norm` | 2 |
| GPU 负载 | `gpu_usage_pct`, `gpu_usage_norm` | 2 |
| GPU 功耗 | `gpu_power_mw`, `delta_gpu_uj` | 2 |
| GPU 延迟 | `gpu_headroom_ms`, `fence_avg_latency_ms` | 2 |
| 帧时间 | `actual_dur_ms`, `lateness_ms`, `frame_dur_norm`, `lateness_norm` | 4 |
| CPU 频率 | `l_cluster_freq_khz`, `m_cluster_freq_khz` + norm | 4 |
| CPU 负载 | `cpu0_usage_pct` ~ `cpu7_usage_pct` + norm | 16 |
| CPU 功耗 | `cpu_power_mw`, `delta_cpu_m_uj`, `delta_cpu_l_uj` | 3 |
| 总功耗 | `total_uj`, `delta_total_uj` | 2 |

**有效状态维度**：约 15–20 维连续特征（选取归一化版本）。

### 1.4 Reward 设计

当前存在 3 套 reward 计算，建议统一为：

```
reward = fps_component + freq_component + power_component

fps_component:
  if fps < 57:  -10 × (57 - fps)     # 大幅惩罚
  elif fps < 59: -2 × (59 - fps)     # 轻微惩罚
  else:          +20                   # 达标奖励

freq_component: (900 - gpu_freq_mhz) / 80   # 低频加分
power_component: (6000 - power_ma) / 400     # 低功耗加分
```

### 1.5 已有数据现状

| 项目 | 状态 |
|------|------|
| 采集脚本 | `collect_qgtf_dataset.py`，QGTF 日志驱动 |
| 已采集轮数 | ~30 轮（action_id s00–s29），数据仍在采集中 |
| 每轮帧数 | ~6,700 帧（约 30 秒采集） |
| 动作覆盖率 | 30/192 = **15.6%** |
| 数据格式 | 逐帧 CSV，含完整状态 + action_id + reward |
| **关键限制** | **每轮 action 固定（轮内不切换）** |

### 1.6 时间依赖性分析（MDP vs Bandit）

| 时间依赖来源 | 机制 | 影响程度 |
|-------------|------|---------|
| 热积累 | 高频率→发热→thermal throttle→未来可用频率降低 | **强** |
| DCVS governor 惯性 | 内部 step counter / penalty accumulator 有记忆 | **中** |
| 游戏场景切换 | 战斗→待机→加载，GPU 负载剧变 | **强** |
| 功耗预算 | 持续高功耗→电池策略介入/温控降频 | **中** |

**结论**：存在非平凡的时间依赖性，MDP 建模比 contextual bandit 更完整。但依赖强度属于"中等"——DCVS 参数不直接改变游戏状态转移，只影响 GPU 对负载的响应方式。

---

## 2. 候选算法评估

### 2.1 Discrete CQL（Conservative Q-Learning）

- **论文**: Kumar et al., "Conservative Q-Learning for Offline Reinforcement Learning", NeurIPS 2020, arXiv:2006.04779
- **核心思想**: 在标准 Bellman error 上添加 Q 值正则项，对数据分布外的 action 施加保守惩罚

$$\mathcal{L}_{\text{CQL}} = \alpha \cdot \mathbb{E}_s\left[\log \sum_a \exp Q(s,a) - \mathbb{E}_{a \sim \hat{\pi}_\beta}[Q(s,a)]\right] + \frac{1}{2}\mathbb{E}_{(s,a,r,s')}[(Q - \mathcal{B}\hat{Q})^2]$$

| 评估维度 | 评分 | 说明 |
|---------|------|------|
| 动作空间匹配 | ★★★★★ | 192 离散 action_id 天然适配 |
| 数据适配性 | ★★★★☆ | 可直接用现有 CSV 构建 (s,a,r,s')，但需改采集加轮内切换 |
| 安全性 | ★★★★★ | 保守估计避免高估未见 action，手机场景零容错最关键 |
| 实现复杂度 | ★★★★☆ | 标准 DQN + 一个正则项，d3rlp/CleanRL 有现成实现 |
| 部署推理延迟 | ★★★★★ | `argmax Q(s,a)` 一次 MLP forward，<1ms |
| offline→online | ★★★☆☆ | 可做但保守性在 online fine-tune 时需要退火 α |
| 数据覆盖不足应对 | ★★★★☆ | 未见 action 的 Q 值被压低，策略偏保守但安全 |

**适用场景**: 首版实现、离散动作空间、安全性优先  
**风险**: 数据覆盖不足（30/192）可能导致策略过度保守

---

### 2.2 IQL（Implicit Q-Learning）

- **论文**: Kostrikov et al., "Offline Reinforcement Learning with Implicit Q-Learning", ICLR 2022, arXiv:2110.06169
- **核心思想**: 用 expectile regression 隐式学习策略，完全不对 OOD action 做 Q 评估

$$\mathcal{L}_V = \mathbb{E}_{(s,a) \sim D}\left[L_\tau^2(Q_{\hat{\theta}}(s,a) - V_\psi(s))\right], \quad L_\tau^2(u) = |\tau - \mathbf{1}(u < 0)| \cdot u^2$$

| 评估维度 | 评分 | 说明 |
|---------|------|------|
| 动作空间匹配 | ★★★★☆ | 原生偏连续空间设计，离散可用但不如 CQL 直接 |
| 数据适配性 | ★★★★★ | 对稀疏数据覆盖最鲁棒——不评估数据外 action |
| 安全性 | ★★★★★ | 策略不会选数据中未出现的 action，极度保守 |
| 实现复杂度 | ★★★☆☆ | 需同时训练 Q/V/π 三个网络 |
| 部署推理延迟 | ★★★★☆ | 需要 policy network forward |
| offline→online | ★★★★★ | **最大优势**——无缝过渡，online 阶段直接继续更新 |
| 扩展性 | ★★★★★ | 可平滑迁移到连续动作空间 |

**适用场景**: 数据覆盖长期不足、需要 offline→online 闭环、未来扩展连续动作  
**优势**: 对 30/192 的低覆盖率最鲁棒

---

### 2.3 TD3+BC（Twin Delayed DDPG + Behavior Cloning）

- **论文**: Fujimoto & Gu, "A Minimalist Approach to Offline Reinforcement Learning", NeurIPS 2021, arXiv:2106.06860
- **核心思想**: TD3 + 行为克隆正则项，实现最简洁的 offline RL

$$\pi = \arg\max_\pi \mathbb{E}_{(s,a) \sim D}\left[\lambda Q(s, \pi(s)) - (\pi(s) - a)^2\right]$$

| 评估维度 | 评分 | 说明 |
|---------|------|------|
| 动作空间匹配 | ★★☆☆☆ | **需要将动作空间改为连续**（如 fsd∈[3,25], pd∈[85,98]） |
| 工程改造量 | ★★☆☆☆ | 需改 action encoding、sysfs 写入逻辑（连续值取整截断） |
| 实现简洁度 | ★★★★★ | 代码量最少的 offline RL 方法 |
| 性能上限 | ★★★★☆ | 连续空间理论上能找到 192 种组合之间的"间隙最优" |
| 当前数据适配 | ★☆☆☆☆ | 现有 192 离散 action 数据不适合连续方法训练 |

**适用场景**: 未来扩展到连续动作空间后  
**当前评估**: 改造成本高于收益，不建议作为第一版

---

### 2.4 Decision Transformer

- **论文**: Chen et al., "Decision Transformer: Reinforcement Learning via Sequence Modeling", NeurIPS 2021, arXiv:2106.01345
- **核心思想**: 将 RL 建模为序列预测问题，用 Transformer 条件生成 action

$$a_t = \text{Transformer}(\hat{R}_t, s_t, \hat{R}_{t-1}, s_{t-1}, a_{t-1}, \ldots)$$

| 评估维度 | 评分 | 说明 |
|---------|------|------|
| 数据需求 | ★☆☆☆☆ | 需要大量长轨迹、多场景数据，当前 30 轮远不够 |
| 长时依赖建模 | ★★★★★ | 热惯性、场景切换的长时依赖是它的专长 |
| Return conditioning | ★★★★☆ | 可指定"给我 FPS=60 且 power<3000mA"条件生成 |
| 实现复杂度 | ★★☆☆☆ | GPT-2 backbone + 训练 infra，比 Q-learning 系重得多 |
| 推理延迟 | ★★☆☆☆ | Transformer 推理比 MLP Q-net 慢 10–50x |
| 工程成熟度 | ★★☆☆☆ | 社区评价两极，很多 benchmark 上不如 CQL/IQL |

**适用场景**: 积累 1000+ 轮跨游戏、跨温度、跨设备数据后的第二代方案  
**当前评估**: 完全不适合首发

---

### 2.5 在线 PPO（Proximal Policy Optimization）

- **论文**: Schulman et al., "Proximal Policy Optimization Algorithms", arXiv:1707.06347
- **评估结论**: **不推荐**

| 评估维度 | 评分 | 说明 |
|---------|------|------|
| 样本效率 | ★☆☆☆☆ | On-policy，每次更新后数据作废，需 10⁵–10⁶ 步交互 |
| 安全性 | ★★☆☆☆ | 初期随机性高，频繁选到糟糕参数导致 FPS 暴跌 |
| 离线数据利用 | ★☆☆☆☆ | 无法利用已有 QGTF 采集数据 |
| 时间成本 | ★☆☆☆☆ | 按每轮 30s 计算，需数千小时真机运行 |

---

### 2.6 在线 DQN（现有实现）

- **现有代码**: `rl_gpu_tuner.py`（tabular Q-learning，非 deep）
- **评估**: 可作为 baseline 但非最优选择

| 评估维度 | 评分 | 说明 |
|---------|------|------|
| 状态表示 | ★★☆☆☆ | 当前用 (fps_bin, freq_bin) 离散化，丢失大量信息 |
| 样本效率 | ★★☆☆☆ | 在线 Q-learning 需要大量交互 |
| 离线数据利用 | ★★☆☆☆ | 标准 DQN 可以用 replay buffer，但无 distributional shift 修正 |
| 安全性 | ★★★☆☆ | ε-greedy 探索有一定风险 |

---

## 3. 最终推荐排名

| 排名 | 算法 | 推荐度 | 核心理由 |
|------|------|--------|---------|
| **1** | **Discrete CQL** | ★★★★★ | 离散动作直接适配 + 保守估计保安全 + 实现成熟 + 推理极快 |
| **2** | **IQL** | ★★★★☆ | 数据覆盖差时最鲁棒 + offline→online 无缝 + 可扩展连续动作 |
| **3** | TD3+BC | ★★★☆☆ | 需连续化改造，当前 ROI 不高，留作连续动作扩展选项 |
| **4** | Decision Transformer | ★★☆☆☆ | 数据量不够、推理慢、工程重，等二代数据充足后考虑 |
| **5** | 在线 PPO | ★☆☆☆☆ | 样本效率极低、探索不安全、不能用离线数据 |

---

## 4. 推荐实施路径

### Phase 1: 数据采集改造

**目标**: 让现有采集数据具备 MDP 转移结构

- 改造 `collect_qgtf_dataset.py`，在每轮 benchmark 内每 5–10 秒切换 DCVS 参数
- 扩大 action 覆盖率至 64+ / 192
- 统一 reward 函数：从 CSV 原始特征重新计算，不依赖 `rl_parse.py` 预计算

### Phase 2: 离线训练（Discrete CQL 首选）

1. **Replay Buffer 构建**: 读取所有 CSV → 构造 $(s_t, a_t, r_t, s_{t+1}, \text{done})$ 元组
2. **状态编码**: 选取 ~15 维归一化特征作为连续状态向量
3. **网络结构**: MLP `state_dim → 256 → 256 → 192`
4. **CQL 训练**: 用 d3rlp 或自实现，关键超参 $\alpha$ 控制保守程度
5. **评估**: 离线策略评估（OPE）+ 少量真机验证

### Phase 3: 部署与在线适应

1. 导出 Q-network 权重
2. 嵌入 `rl_gpu_tuner.py` 做实时推理: 每 N 帧观测状态 → `argmax Q(s,a)` → 写 sysfs
3. 安全阀: 如果 FPS 连续低于 57 → 回退默认参数
4. 可选: 少量 online fine-tune（需逐步退火 CQL α）

### Fallback: 如果 CQL 过于保守

切换到 **IQL** — 它对数据覆盖不足的鲁棒性更好，且 offline→online 过渡更自然。

---

## 5. 关键参考文献

| 算法 | 论文 | 年份/会议 | arXiv |
|------|------|----------|-------|
| CQL | Kumar et al., "Conservative Q-Learning for Offline RL" | NeurIPS 2020 | 2006.04779 |
| IQL | Kostrikov et al., "Offline RL with Implicit Q-Learning" | ICLR 2022 | 2110.06169 |
| TD3+BC | Fujimoto & Gu, "A Minimalist Approach to Offline RL" | NeurIPS 2021 | 2106.06860 |
| Decision Transformer | Chen et al., "RL via Sequence Modeling" | NeurIPS 2021 | 2106.01345 |
| PPO | Schulman et al., "Proximal Policy Optimization" | arXiv 2017 | 1707.06347 |
| DVFO (DVFS+DRL) | Zhang et al., "Learning-Based DVFS for Edge-Cloud Inference" | arXiv 2023 | 2306.01811 |
| Bayesian Optimization | Frazier, "A Tutorial on Bayesian Optimization" | arXiv 2018 | 1807.02811 |

---

## 6. 工具与依赖

| 工具 | 用途 |
|------|------|
| [d3rlp](https://github.com/takuseno/d3rlpy) | CQL / IQL / TD3+BC 开箱即用实现 |
| PyTorch | 神经网络训练 |
| numpy / pandas | 数据处理与 replay buffer 构建 |
| BoTorch / scikit-optimize | 贝叶斯优化（如需对比 baseline） |

---

## 7. 备注

- 本文档由 **GitHub Copilot (Claude Opus 4.6)** 基于项目代码和数据分析生成
- 分析基于 2026-04-01 时项目状态：30/192 action 已采集，数据仍在持续收集中
- 建议结合其他 AI 分析结果做最终综合决策
- 数据采集改造（轮内动作切换）是所有 RL 方法的共同前提
