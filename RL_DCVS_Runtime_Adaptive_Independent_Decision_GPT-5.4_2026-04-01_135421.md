# RL-DCVS 运行时自适应模型独立决策纪要

> **文档性质**：独立决策备忘录
>
> **用途**：供后续与其他 AI 结论做综合对比时使用
>
> **生成时间（UTC）**：2026-04-01 13:54:21
>
> **生成工具**：GitHub Copilot CLI
>
> **模型名称**：GPT-5.4
>
> **模型 ID**：`gpt-5.4`
>
> **信息来源范围**：仅基于本次对话、我对项目 `/mnt/d/QCOM-RL` 的代码/结构分析，以及本次对话中提到的公开论文和开源项目
>
> **排除项**：**不使用**任何目标目录中其他 AI 已写文档的内容；本文件是独立形成的判断

---

## 1. 问题背景

你的目标不是寻找一组固定的“静态最优” DCVS 参数，而是：

- 在**手机运行时**根据当前负载和设备状态，**实时自适应**调整 DCVS 参数；
- 在保证目标帧率/帧率稳定性的前提下，动态平衡**性能、功耗、频率**；
- 核心是**动态最优**，而不是单次 benchmark 后的固定最优组合。

结合项目现状，这个仓库已经具备比较完整的工程闭环：

- `rl_gpu_tuner.py`：当前在线控制器，使用**表格 Q-learning**，按时间窗口改参数；
- `collect_offline_dataset.py`、`collect_qgtf_dataset.py`：离线数据采集链路，能生成 `-RL.bin / -RL.csv`；
- `baseline_benchmark.py`、`compare_baseline.py`：基线与效果对比工具；
- `HOK/rl_gpu_tuner.py`：更工程化的版本，已纳入更多监控信号，如 CPU 频率、外部功耗输入等；
- `tool/csv_plotly_reviewer.py` 显示离线 CSV 已经具备更丰富的逐帧特征：`gpu_usage_pct`、`actual_dur_ms`、`lateness_ms`、`gpu_power_mw`、`delta_total_uj`、`fence_avg_latency_ms`、`action_id`、归一化特征等。

这说明：

**项目真正缺少的不是“采集数据/写参数”的能力，而是更适合运行时动态决策的模型。**

---

## 2. 我对问题建模的判断

### 2.1 这不是纯静态黑盒优化问题

如果目标只是：

- 固定场景；
- 固定一轮 benchmark；
- 给定 4 个 DCVS 参数组合；
- 找到一组 reward 最好的常量配置；

那么问题非常接近**静态黑盒优化**，Bayesian Optimization（BO）会很合适。

但你的目标明确是：

- **运行时自适应**；
- 根据**当前上下文**动态找最优动作；
- 实时调参；
- 不考虑静态最优。

这会把问题从“全局静态最优搜索”变成：

> **给定当前上下文 `c_t`，从当前动作集合 `a_t` 中选择一个最优动作，使即时 reward 最大，同时满足性能/热/功耗安全约束。**

因此，纯静态 BO 不能作为主模型，只能作为辅助工具。

### 2.2 这也不是必须优先上完整长时序 RL 的问题

重新评估后，我的判断是：

- `DCVS` 参数确实会通过**频率轨迹、热积累、governor hysteresis、场景切换**影响后续窗口；
- 所以它**不是**完全理想化的“action 不影响 next state”的无记忆 bandit；
- 但从工程角度看，这种影响更像是**短期滞后和缓慢漂移**，而不是必须用 PPO/SAC/DQN 这类完整 online RL 才能处理的强 MDP。

因此，更合理的建模是：

> **带短历史特征的运行时上下文决策问题（short-history contextual decision）**

也就是：

- 主体用 **Contextual Bandit / Contextual BO** 处理“当前上下文 -> 当前最优动作”；
- 把过去 1~3 个窗口的统计量并入 context，吸收短期热惯性和 governor 记忆；
- 再通过**安全约束、回退逻辑、最小驻留时间**弥补 bandit 的无记忆缺陷。

这比直接上完整 PPO/SAC/DQN 更现实，也比纯静态 BO 更贴近你的目标。

---

## 3. 候选方法重新评估

下面的排序，是**围绕“运行时自适应、动态最优、实时调参”**这个目标重新给出的。

### 3.1 Safe Contextual Bandit（首选）

**推荐度：9.5 / 10**

推荐形式：

- 第一选择：**Thompson Sampling over discrete actions**
- 可解释基线：**LinUCB**
- 非线性增强版：**Neural Contextual Bandit / Neural Thompson Sampling**

#### 为什么排第 1

1. **最贴合当前目标**
   - 你的目标是 runtime adaptive，而不是静态全局搜索。
   - 当前更像“当前上下文下选当前最优动作”，这正是 contextual bandit 的主场。

2. **最贴合当前动作空间**
   - 仓库当前动作空间本质上是离散 DCVS 参数组合。
   - 对离散动作，bandit 非常自然，不需要强行连续化。

3. **样本效率高，真机代价低**
   - 手机在线探索代价高，bandit 比 PPO/SAC/DQN 更省样本。
   - 它只优化当前动作，不做复杂的 value propagation。

4. **部署简单、推理快、容易加安全壳**
   - 每个决策窗口只需计算当前 context 下各动作的不确定性和期望回报。
   - 很适合加入 guardrail：低 FPS 禁止激进降频，高温时只允许保守动作。

5. **能直接利用你现有工程链路**
   - 当前仓库已能拿到 FPS、GPU freq、GPU busy、power，`HOK/` 版还有 CPU 频率和外部功耗 feed。
   - 离线 CSV 也具备更丰富的上下文特征，可直接用于离线初始化和在线校准。

#### 局限

- 纯 bandit 假设“当前动作不重要地影响未来”，这在 DVFS 场景里并不完全成立。
- 但这个问题可以通过以下方式缓解：
  - 把**最近 1~3 个窗口历史**并入 context；
  - 增加**最小驻留时间**；
  - 限制动作跳变幅度；
  - 配套安全回退逻辑。

所以我的最终结论不是“纯无记忆 bandit”，而是：

> **Safe Short-History Contextual Bandit**

这是我认为最适合当前项目的首发方案。

---

### 3.2 Contextual Bayesian Optimization（次优，但非常强）

**推荐度：8.5 / 10**

#### 为什么它很强

你的建模思路成立：

- `context c_t = (GPU workload, thermal state, game scene, ...)`
- `action a_t = (fsd, pd, pu, sf)`
- `reward r_t = f(FPS, power, freq, stability)`

如果把 reward 看成一个带上下文的黑盒函数：

> `r = f(context, action)`

那么 contextual BO 完全合理。

它的优点：

- 比纯 RL 样本效率高很多；
- 天然处理噪声；
- 适合在线逐步试探；
- 如果后续动作空间改成连续 residual 调整，会更强。

#### 为什么仍排在 bandit 后面

1. **当前动作空间更偏离散**
   - 你当前不是在真正连续的 4 维控制空间里做细粒度调节，而是在一组有限 DCVS 参数组合里选参。
   - 在这种情况下，bandit 往往比 GP-BO 更直接。

2. **上下文维度会快速膨胀**
   - 一旦把 `fps_error`、`1% low`、`gpu_busy`、`cpu/gpu freq`、`power`、`temperature`、`jank/lateness`、`scene`、`prev action` 等都放进去，普通 GP 会变得较重，调参也更麻烦。

3. **工程可控性上，bandit 更轻量**
   - 运行时维护 GP、计算 acquisition、处理分类/离散动作映射，复杂度更高。
   - 如果目标是“尽快在真机上稳定跑起来”，bandit 更务实。

#### 定位

- **不是第 1 选择，但非常值得保留为第 2 选择**；
- 如果后续把动作空间改成：
  - 连续 residual 调整；
  - 联调 CPU/GPU/thermal headroom；
  - 低维连续控制；

那么 contextual BO 的地位会显著上升，甚至可能反超 bandit。

---

### 3.3 Safe Offline-to-Online RL（IQL / CQL warm start）

**推荐度：7 / 10**

这里不是指“纯在线 RL 首发”，而是：

- 先用离线数据做保守初始化；
- 再在线小步更新；
- 用 IQL / CQL 一类方法做 warm start。

#### 为什么不是第 1

1. **当前主需求不是长时序 credit assignment**
   - 你现在更关心“当前上下文下立刻选对动作”。
   - 这更偏 bandit / contextual optimization，而不是必须用 RL 的 delayed return。

2. **在线 RL 的风险和复杂度更高**
   - 手机真机探索容错低。
   - 离线到在线的策略漂移、分布偏移、更新稳定性都需要额外处理。

3. **当前工程状态更适合轻量控制器优先**
   - 先跑通安全的运行时自适应，再判断是否真的需要 full RL。

#### 何时优先级会上升

如果后续验证发现：

- 热积累和温控回授对未来 30~120 秒表现影响很大；
- 当前动作对后续多个窗口的可行域影响显著；
- 开始做 CPU+GPU+thermal 联合控制；
- 需要更强的 delayed reward 建模；

那 IQL/CQL warm start 的价值会显著上升。

#### 结论

- **可作为第二阶段/增强方案**；
- **不建议作为当前首发主控制器**。

---

### 3.4 静态 Bayesian Optimization（BO）

**推荐度：6 / 10**

你的判断有相当一部分是对的：

- 从样本效率看，BO 很强；
- 从噪声处理看，BO 很合适；
- 从“每轮一个 action + 一个平均 reward”的离线 benchmark 形态看，它非常契合；
- 用来快速找到某类场景下的好参数，它很务实。

#### 但它的核心问题

- 它主要回答的是：
  - “这个场景/这类 workload 下，哪组参数最优？”
- 它**不直接回答**：
  - “当 workload / temperature / scene 在运行时变化时，我应该实时怎么改？”

所以 BO 在你的目标下应该被降级为：

- **离线校准器**；
- **action space pruning 工具**；
- **contextual 方法的先验生成器**；
- **给 bandit / RL 初始候选动作集**。

#### 定位

> BO 很实用，但更适合当“辅助工具”，不适合做最终主模型。

---

### 3.5 当前表格 Q-learning / 传统在线 DQN

**推荐度：4.5 / 10**

当前仓库里的 `rl_gpu_tuner.py` / `HOK/rl_gpu_tuner.py` 是不错的 baseline，但不适合作为最终方向。

主要问题：

- 状态太粗：主要是 `avg_fps + avg_freq` 离散桶；
- 没有充分利用现有更丰富的 telemetry；
- 在线探索风险高；
- 样本效率一般；
- 不容易在手机真机上安全扩展。

结论：

- **保留为 baseline / fallback**；
- **不建议作为最终模型选择**。

---

### 3.6 PPO / vanilla SAC / 纯在线深度 RL

**推荐度：3 / 10**

不建议作为当前场景首发。

原因：

- 在线探索成本太高；
- 手机真机环境容错低；
- 样本效率不足；
- 安全回退和守护机制复杂；
- 已经有更符合工程需求的上下文决策方法可选。

---

## 4. 最终推荐排序

### 4.1 面向“当前项目 + 立即可做”的排序

| 排名 | 方法 | 推荐度 | 结论 |
|---|---|---:|---|
| **1** | **Safe Short-History Contextual Bandit** | **9.5/10** | 当前最推荐，最贴近 runtime adaptive 目标 |
| **2** | **Contextual Bayesian Optimization** | **8.5/10** | 很强，尤其适合后续连续 residual 调参 |
| **3** | **Safe Offline-to-Online RL（IQL/CQL warm start）** | **7.0/10** | 二阶段方案，长时序效应很强时再提升优先级 |
| **4** | **静态 BO** | **6.0/10** | 很实用，但更适合作为辅助校准器 |
| **5** | **当前表格 Q-learning / DQN** | **4.5/10** | 适合做 baseline，不适合做最终方向 |
| **6** | **PPO / vanilla SAC** | **3.0/10** | 在线探索代价过高，不推荐首发 |

### 4.2 如果未来动作空间改成连续 residual 调参，排序可能变化

如果后面把动作定义从“离散 action_id 选参”改成：

- 对 `first_step_down / penalty_down / penalty_up / strict_frame` 做连续 residual 微调；
- 或扩展到 CPU/GPU 联调；

那么我会把排序改成：

| 排名 | 方法 | 变化后的理由 |
|---|---|---|
| **1** | **Contextual BO** | 连续低维动作下优势更大 |
| **2** | **Safe Neural Contextual Bandit** | 仍然很强，但不再天然优于 BO |
| **3** | **Safe Offline-to-Online RL** | 当长时序和耦合变强时价值上升 |

---

## 5. 最终模型选择

### 最终选择

> **最终推荐模型：Safe Short-History Contextual Bandit**
>
> **首发算法建议：Thompson Sampling over safe discrete action set**
>
> **可解释基线建议：LinUCB**

### 我为什么最终选它

#### 原因 1：最贴合你的核心目标

关键词是：

- 运行时；
- 自适应；
- 动态最优；
- 实时调节；
- 不考虑静态。

在这个目标下，bandit 的建模最贴近“当前看上下文，当前选动作”。

#### 原因 2：最贴合当前项目动作定义

当前项目动作是 DCVS 参数组合，本质上是离散动作集。

- 对离散动作，bandit 很自然；
- 不需要先把动作连续化；
- 不需要一开始就上完整 RL。

#### 原因 3：样本效率和安全性平衡最好

- 比纯 online RL 更省真机试验成本；
- 比 contextual BO 更轻量；
- 比静态 BO 更符合 runtime adaptive；
- 比当前 Q-learning 更容易做工程化安全壳。

#### 原因 4：能利用现有 telemetry 和 richer offline logs

建议把当前和最近短历史窗口特征作为 context，包括：

- `fps`
- `fps_error = fps - target`
- `1% low / min fps`
- `gpu_busy`
- `gpu_freq`
- `cpu0/cpu6 freq`
- `power_ma` / 外部功耗
- `actual_dur_ms`
- `lateness_ms`
- `gpu_headroom_ms`
- `fence_avg_latency_ms`
- `delta_total_uj`
- `scene label`
- `temperature / thermal headroom`（如果可获取）
- `prev_action`
- `prev_reward`
- 最近 2~3 个窗口的趋势统计

#### 原因 5：容易做“安全优先”的上线策略

最终部署时建议不是“所有动作都可选”，而是：

1. 先根据 guardrail 过滤动作；
2. 再用 contextual bandit 在安全动作里选最优；
3. 若出现异常，立即回退。

这非常适合手机运行时调参场景。

---

## 6. 我建议的运行时控制框架

### 6.1 控制周期

建议每 **3~10 秒**做一次决策，不建议每帧都调。

原因：

- 太短会放大噪声，造成抖动；
- 太长会降低自适应能力；
- 3~10 秒更适合吸收 workload/thermal 的短期变化。

### 6.2 决策逻辑

```text
采集当前窗口统计
    -> 构造 context（当前 + 短历史）
    -> 安全过滤候选动作
    -> Contextual Bandit 选动作
    -> 写入 DCVS 参数
    -> 观察 reward / guardrail violation
    -> 更新 bandit
```

### 6.3 建议的 reward 原则

不建议继续用单纯线性的“低频加分 + 低功耗加分”作为唯一目标，而建议改成：

> **约束优先 + 可行域内优化**

也就是：

- 第一优先级：保证 `FPS >= target`；
- 第二优先级：避免 `1% low`、`jank`、`lateness` 明显恶化；
- 第三优先级：在满足前两项后，再优化功耗/频率。

换句话说：

- **先保体验，再省电。**

### 6.4 建议的安全机制

至少应包含：

- **FPS guardrail**：若连续窗口低于阈值，禁止激进降频动作；
- **thermal guardrail**：温度/thermal headroom 触线时只允许保守动作；
- **dwell time**：动作应用后至少停留若干窗口，避免来回抖动；
- **step-size limit**：限制动作跳变幅度；
- **rollback**：出现严重性能回退时立即退回默认/保守参数；
- **safe action subset**：只在经过离线验证的一组动作内在线探索。

---

## 7. 各类方法在这个项目里的合理定位

### 7.1 Safe Contextual Bandit

**主控制器**。

### 7.2 Contextual BO

**备选主控制器 / 连续动作升级路线**。

### 7.3 静态 BO

**辅助工具**，用于：

- 初始动作筛选；
- 为 bandit 提供初始 prior；
- 离线 benchmark 校准；
- 降低在线搜索空间。

### 7.4 IQL / CQL

**第二阶段增强方案**，用于：

- 更强的长时序建模；
- offline-to-online warm start；
- 联合控制 CPU/GPU/thermal 的更复杂版本。

---

## 8. 参考论文与开源项目

### 8.1 与 BO / Contextual BO 相关

- Peter Frazier, **A Tutorial on Bayesian Optimization**, arXiv:1807.02811

### 8.2 与 Offline RL / Safe warm start 相关

- Kumar et al., **Conservative Q-Learning**, arXiv:2006.04779
- Kostrikov et al., **Implicit Q-Learning**, arXiv:2110.06169
- Fujimoto & Gu, **TD3+BC**, arXiv:2106.06860
- Chen et al., **Decision Transformer**, arXiv:2106.01345

### 8.3 与工程实现相关

- `takuseno/d3rlpy`：支持 CQL / IQL / TD3+BC / Decision Transformer
- `ztt-21/zTT`：移动端 DVFS + RL 的参考工程，适合理解 client-agent 和热/功耗联合控制思路
- `BoTorch` / `Ax`：BO / Contextual BO 工具链

---

## 9. 最终一句话结论

如果目标是：

- **手机运行时自适应**；
- **实时动态找当前最优**；
- **动作仍是当前这类离散 DCVS 参数组合**；
- **希望尽快做出可上线、可控、可回退的方案**；

那么我的最终建议是：

> **首发主模型选 `Safe Short-History Contextual Bandit`；**
>
> **静态 BO 用作辅助校准；**
>
> **Contextual BO 作为连续动作升级路线；**
>
> **IQL/CQL warm start 作为二阶段增强，而不是当前首发主控制器。**

---

## 10. 备注

- 本文件是我基于本次对话和项目分析形成的**独立判断**；
- 它的定位是：帮助你后续与其他 AI 的结论做**横向对比和综合决策**；
- 如果后续决定把动作空间改成连续 residual 调参，我会建议重新评估 `Contextual BO` 与 `Neural Contextual Bandit` 的优先级。
