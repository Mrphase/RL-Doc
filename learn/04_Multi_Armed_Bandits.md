# 04｜多臂老虎机：先学最简单的在线决策框架

## 前置阅读

- 建议先读 `03_Exploration_vs_Exploitation.md`，先把“探索”和“利用”的拉扯感建立起来。
- 如果你还没读前文，也没关系，这一篇会从零讲起，并把 DCVS 背景一起带上。
- 读这篇时，脑子里一直记住一个工程事实：我们面对的是 `6×4×4×2 = 192` 个离散动作，
  GPU 频率大致落在 `282-710 MHz`，目标 FPS 常常是 `59-60`，功耗观测范围常见为
  `0-8000 mW`。

## 先抓住直觉：赌场那一排老虎机，像不像 192 个 DCVS 参数组合

想象你走进赌场，面前排着很多台老虎机。每台机器的中奖概率都不一样，但你一开始并不知道。

- 你只有有限的试玩次数；
- 你想在总次数固定的前提下，尽量多赚钱；
- 但你又不能把所有次数都浪费在“看起来很烂”的机器上。

这就是**多臂老虎机（Multi-Armed Bandit）**的核心问题。

把它搬到 GPU DCVS 调参里，画风几乎一模一样：

- 拉一次老虎机摇杆，等价于“给下一个控制窗口选一个 `action_id`”；
- 中奖金额，等价于“这一小段时间拿到的奖励”；
- 奖励高，说明这个动作在当前目标下更划算；
- 奖励低，说明它要么费电，要么掉帧，要么两头都不占。

所以你可以把 192 个 DCVS 参数组合，想成 192 台可选的老虎机。
Bandit 不是完整强化学习里最强的那一类，但它是最容易上手、最容易在线部署，
也最适合拿来建立直觉的第一站。

### ① 直觉类比

你可以把自己想成一个值班调参工程师。系统每隔几秒就问你一次：

- 继续用上次那个稳妥动作？
- 还是试试另一个可能更省电的动作？

如果你永远只选当前最稳的动作，你会很安全，但可能永远学不到更好的配置。
如果你疯狂乱试，你学得很快，但玩家很可能先被你试到掉帧。

### ② 正式定义

在最基础的 Bandit 设定里：

- 一共有 $K$ 个动作，也叫 $K$ 个“臂”；
- 第 $t$ 轮，你选一个动作 $a_t$；
- 系统返回一个即时奖励 $r_t$；
- 每个动作背后都有一个你看不见的真实平均奖励 $\mu_a$。

这里的 $\mu_a$，你可以先把它理解成“动作 $a$ 长期平均能拿多少分”。

$$
a_t \in \{1,2,\dots,K\}
$$

$$
\mu_a = \mathbb{E}[r \mid a]
$$

Bandit 的目标很简单：在有限轮数里，让总奖励尽量大。

如果把前 $T$ 轮总奖励记成 $G_T$，那就是：

$$
G_T = \sum_{t=1}^{T} r_t
$$

### ③ 公式推导 + TikZ 图

这张图展示了什么：下面这张流程图把“选动作 → 运行窗口 → 看结果 → 更新统计”的
最小闭环画出来了。它就是 plain bandit 的日常工作流。

```latex
\begin{tikzpicture}[>=Stealth, node distance=12mm and 10mm]
  \tikzstyle{inputnode}=[draw, rounded corners, fill=blue!15,
    minimum width=34mm, minimum height=9mm, align=center]
  \tikzstyle{processnode}=[draw, rounded corners, fill=orange!18,
    minimum width=36mm, minimum height=9mm, align=center]
  \tikzstyle{outputnode}=[draw, rounded corners, fill=green!18,
    minimum width=36mm, minimum height=9mm, align=center]

  \node[inputnode] (context) {输入\\当前窗口信息\\GPU 负载、温度、FPS};
  \node[processnode, below=of context] (pick) {处理\\从 192 个动作里\\选 1 个 action\_id};
  \node[processnode, below=of pick] (run) {处理\\运行一个控制窗口\\例如 5 秒};
  \node[outputnode, below=of run] (observe) {输出\\观察 FPS、功耗、稳定性};
  \node[processnode, below=of observe] (reward) {处理\\算奖励并更新\\每个动作统计};
  \node[outputnode, below=of reward] (next) {输出\\进入下一轮\\继续选动作};

  \draw[->, thick] (context) -- node[right] {当前状态} (pick);
  \draw[->, thick] (pick) -- node[right] {执行动作} (run);
  \draw[->, thick] (run) -- node[right] {收集结果} (observe);
  \draw[->, thick] (observe) -- node[right] {回写奖励} (reward);
  \draw[->, thick] (reward) -- node[right] {更新后再决策} (next);
\end{tikzpicture}
```

图里的关键路径/要点：Bandit 只盯“这次选什么，立刻拿到什么回报”。
它不像完整 MDP 那样显式建模长时序状态转移，所以它简单，也因此有局限。

### ④ DCVS 实际数值计算示例

先用一个非常朴素、但足够好懂的奖励函数来算一遍。假设一个控制窗口结束后，
我们按下面的方法给这个动作打分：

$$
r_{\text{base}} = \frac{\min(\text{FPS}, 60)}{60} - 0.25 \cdot \frac{\text{功耗}}{8000}
$$

$$
r =
\begin{cases}
r_{\text{base}}, & \text{如果 FPS} \ge 59 \\
r_{\text{base}} - 0.15, & \text{如果 FPS} < 59
\end{cases}
$$

这里的意思很直白：

- FPS 越接近 `60` 越好；
- 功耗越低越好；
- 但如果 FPS 跌破 `59`，就额外扣一笔大分，因为玩家会明显感觉到卡顿。

现在假设我们手上先看 4 个候选动作：

| action_id | GPU 频率 | 观测 FPS | 功耗 | 解释 |
|---|---:|---:|---:|---|
| action_id 12 | `500 MHz` | `60` | `5000 mW` | 很稳，但偏费电 |
| action_id 72 | `545 MHz` | `60` | `4200 mW` | 稳定且比 12 更省 |
| action_id 109 | `430 MHz` | `59` | `3600 mW` | 勉强达标，但更省电 |
| action_id 155 | `355 MHz` | `58` | `3100 mW` | 很省电，但掉帧 |

下面一步一步算。

**action_id 12：**

$$
\frac{\min(60,60)}{60} = \frac{60}{60} = 1
$$

$$
0.25 \cdot \frac{5000}{8000} = 0.25 \cdot 0.625 = 0.15625
$$

$$
r_{12} = 1 - 0.15625 = 0.84375
$$

**action_id 72：**

$$
\frac{\min(60,60)}{60} = 1
$$

$$
0.25 \cdot \frac{4200}{8000} = 0.25 \cdot 0.525 = 0.13125
$$

$$
r_{72} = 1 - 0.13125 = 0.86875
$$

**action_id 109：**

$$
\frac{\min(59,60)}{60} = \frac{59}{60} = 0.9833333
$$

$$
0.25 \cdot \frac{3600}{8000} = 0.25 \cdot 0.45 = 0.1125
$$

$$
r_{109} = 0.9833333 - 0.1125 = 0.8708333
$$

因为它刚好还在 `59-60` 目标带里，所以这里不触发额外扣分。

**action_id 155：**

$$
\frac{\min(58,60)}{60} = \frac{58}{60} = 0.9666667
$$

$$
0.25 \cdot \frac{3100}{8000} = 0.25 \cdot 0.3875 = 0.096875
$$

$$
r_{\text{base},155} = 0.9666667 - 0.096875 = 0.8697917
$$

$$
r_{155} = 0.8697917 - 0.15 = 0.7197917
$$

这一轮里，4 个动作的结果是：

- action_id 109：$0.8708333$
- action_id 72：$0.86875$
- action_id 12：$0.84375$
- action_id 155：$0.7197917$

你会发现：Bandit 真正在学的，不是“绝对真理”，而是“谁在当前目标函数下更值钱”。

## 1. 累积遗憾：不是看你赚了多少，而是看你错过了多少

### ① 直觉类比

假设赌场里最赚钱的那台老虎机，你事后才知道原来是 3 号机。
那你前面去拉 1 号机、2 号机、4 号机的那些回合，
就相当于“错过了本来可以拿到的更高回报”。

这部分“本来能赚到，但没赚到”的差额，就是遗憾（Regret）。

### ② 正式定义

记最优动作的真实平均奖励为：

$$
\mu^* = \max_a \mu_a
$$

那么前 $T$ 轮的**累积遗憾（Cumulative Regret）**定义为：

$$
R_T = \sum_{t=1}^{T} (\mu^* - \mu_{a_t})
$$

它的意思是：每一轮都把“这轮本来能拿到的最好平均奖励”
和“你实际选中动作的平均奖励”做差，再把这些差额全部累加起来。

### ③ 公式推导 + TikZ 图

这张图展示了什么：下面这张图把“最佳动作”和“本轮所选动作”的差额，
如何一轮一轮累加成累积遗憾，画成了一条流水线。

```latex
\begin{tikzpicture}[>=Stealth, node distance=12mm and 10mm]
  \tikzstyle{inputnode}=[draw, rounded corners, fill=blue!15,
    minimum width=32mm, minimum height=9mm, align=center]
  \tikzstyle{processnode}=[draw, rounded corners, fill=orange!18,
    minimum width=34mm, minimum height=9mm, align=center]
  \tikzstyle{outputnode}=[draw, rounded corners, fill=green!18,
    minimum width=34mm, minimum height=9mm, align=center]

  \node[inputnode] (best) {输入\\最优动作均值\\$\mu^*$};
  \node[inputnode, right=25mm of best] (chosen) {输入\\本轮所选动作均值\\$\mu_{a_t}$};
  \node[processnode, below=of best, xshift=12mm] (gap) {处理\\先算单轮差额\\$\mu^* - \mu_{a_t}$};
  \node[outputnode, below=of gap] (sum) {输出\\把每轮差额累加\\得到 $R_T$};

  \draw[->, thick] (best) -- node[left] {最佳参考} (gap);
  \draw[->, thick] (chosen) -- node[right] {实际选择} (gap);
  \draw[->, thick] (gap) -- node[right] {逐轮累加} (sum);
\end{tikzpicture}
```

图里的关键路径/要点：遗憾不是说“你这轮一定失败了”，
而是说“你这轮距离最好答案还差了多少”。

### ④ DCVS 实际数值计算示例

沿用上面 4 个动作的均值，当前最优动作是 `action_id 109`，
所以：

$$
\mu^* = 0.8708333
$$

假设前 5 轮你实际选动作的顺序是：

- 第 1 轮：action_id 12
- 第 2 轮：action_id 72
- 第 3 轮：action_id 155
- 第 4 轮：action_id 109
- 第 5 轮：action_id 72

那每一轮的遗憾分别是：

$$
\text{第 1 轮遗憾} = 0.8708333 - 0.84375 = 0.0270833
$$

$$
\text{第 2 轮遗憾} = 0.8708333 - 0.86875 = 0.0020833
$$

$$
\text{第 3 轮遗憾} = 0.8708333 - 0.7197917 = 0.1510416
$$

$$
\text{第 4 轮遗憾} = 0.8708333 - 0.8708333 = 0
$$

$$
\text{第 5 轮遗憾} = 0.8708333 - 0.86875 = 0.0020833
$$

$$
R_5 = 0.0270833 + 0.0020833 + 0.1510416 + 0 + 0.0020833 = 0.1822915
$$

这个数越小，说明你越快摸到了好动作。
所以很多 Bandit 算法表面上在“选动作”，本质上都在想办法把遗憾压低。

## 2. 为什么“多拉几次”会更准：样本均值与大数定律

### ① 直觉类比

如果一家餐馆你只吃过 1 次，你很难说它到底稳不稳。
但如果你吃了 50 次，而且大部分时候都不错，
你对它的判断就会越来越有底。

Bandit 里对每个动作的认识，也是在靠“试的次数”慢慢变准。

### ② 正式定义

动作 $a$ 被试了 $N_a$ 次之后，它的**样本均值**通常写成：

$$
\hat{\mu}_a = \frac{1}{N_a} \sum_{i=1}^{N_a} r_i(a)
$$

这里：

- $N_a$ 是这个动作被选中的次数；
- $r_i(a)$ 是第 $i$ 次选这个动作时观察到的奖励；
- $\hat{\mu}_a$ 是我们当前对它的“经验平均分”。

> **补充知识：样本均值和大数定律**
>
> 样本均值，你可以直接理解成“把已经看到的分数全部加起来，再除以次数”。
> 它一点都不神秘，本质上就是求平均数。
>
> 大数定律（Law of Large Numbers）的意思是：如果同一件事反复做很多次，
> 那它的平均结果会越来越接近真实期望。
> 在 Bandit 里就是：某个动作被试得越多，$\hat{\mu}_a$ 通常就越接近它真正的 $\mu_a$。
> 这就是为什么“没试够的动作”天然带着不确定性。

### ③ 公式推导 + TikZ 图

这张图展示了什么：下面这张图把“奖励序列 → 求和 → 除以次数 → 得到经验均值”
的步骤拆开了。

```latex
\begin{tikzpicture}[>=Stealth, node distance=12mm and 10mm]
  \tikzstyle{inputnode}=[draw, rounded corners, fill=blue!15,
    minimum width=34mm, minimum height=9mm, align=center]
  \tikzstyle{processnode}=[draw, rounded corners, fill=orange!18,
    minimum width=34mm, minimum height=9mm, align=center]
  \tikzstyle{outputnode}=[draw, rounded corners, fill=green!18,
    minimum width=34mm, minimum height=9mm, align=center]

  \node[inputnode] (seq) {输入\\奖励序列\\$0.87, 0.86, 0.88, 0.87$};
  \node[processnode, below=of seq] (sum) {处理\\先求和\\$0.87+0.86+0.88+0.87$};
  \node[processnode, below=of sum] (div) {处理\\再除以次数\\$\div 4$};
  \node[outputnode, below=of div] (mean) {输出\\样本均值\\$\hat{\mu}=0.87$};

  \draw[->, thick] (seq) -- node[right] {收集数据} (sum);
  \draw[->, thick] (sum) -- node[right] {平均化} (div);
  \draw[->, thick] (div) -- node[right] {得到估计} (mean);
\end{tikzpicture}
```

图里的关键路径/要点：Bandit 最常维护的统计量，就是“次数”和“均值”。
后面 UCB、Thompson Sampling，都是在这两个量上继续加工。

### ④ DCVS 实际数值计算示例

假设 `action_id 72` 在过去 4 次窗口里的奖励依次是：

- 第 1 次：$0.87$
- 第 2 次：$0.86$
- 第 3 次：$0.88$
- 第 4 次：$0.87$

那它的样本均值就是：

$$
\hat{\mu}_{72} = \frac{0.87 + 0.86 + 0.88 + 0.87}{4}
$$

先算分子：

$$
0.87 + 0.86 + 0.88 + 0.87 = 3.48
$$

再除以 4：

$$
\hat{\mu}_{72} = 3.48 \div 4 = 0.87
$$

再看 `action_id 109`，如果你目前只观测过两次，奖励是 $0.89$ 和 $0.86$，
那么：

$$
\hat{\mu}_{109} = \frac{0.89 + 0.86}{2} = \frac{1.75}{2} = 0.875
$$

表面上看，`action_id 109` 的均值比 `action_id 72` 高。
但它只试了 2 次，明显还没有 72 那么“心里有底”。
UCB 正是冲着这个问题来的。

## 3. 置信上界（Upper Confidence Bound, UCB）：均值不够，还要加探索奖金

### ① 直觉类比

想象你在逛夜市，已经反复吃过的摊位，口味你很清楚；
只吃过 1 次的新摊位，就算那次表现一般，你也不会马上把它彻底判死刑。

UCB 的想法就是：

- 已经试很多次的动作，主要看经验均值；
- 还没试够的动作，额外给一点“探索奖金”；
- 奖金会随着试的次数增加而变小。

### ② 正式定义

UCB 常见写法是：

$$
a_t = \arg\max_a \left[\hat{\mu}_a + c\sqrt{\frac{\ln t}{N_a}}\right]
$$

你可以把它拆成两部分：

$$
\text{UCB}_a(t) = \hat{\mu}_a + c\sqrt{\frac{\ln t}{N_a}}
$$

第一项 $\hat{\mu}_a$ 是“目前看起来有多好”，
第二项 $c\sqrt{\frac{\ln t}{N_a}}$ 是“我还该不该多给你一点试用机会”。

每一项分别是什么意思：

- $\hat{\mu}_a$：动作 $a$ 当前的经验均值；
- $N_a$：动作 $a$ 已经被试过多少次；
- $t$：当前已经进行到第几轮；
- $\ln t$：让探索压力随着总轮数增加而缓慢上升；
- $c$：人为控制探索力度的系数。

如果你想要一句最短解释，那就是：

$$
\text{UCB 分数} = \text{经验均值} + \text{不确定性奖金}
$$

### ③ 公式推导 + TikZ 图

这张图展示了什么：下面这张图把 4 个动作的“样本均值 + 置信区间”画成了柱状图。
蓝色柱子表示经验均值，橙色误差线表示不确定性，绿色箭头标出当前 UCB 最高的动作。

```latex
\begin{tikzpicture}[x=1.9cm,y=4.0cm,>=Stealth]
  \tikzstyle{inputnode}=[draw, rounded corners, fill=blue!15,
    minimum width=22mm, minimum height=8mm, align=center]
  \tikzstyle{processnode}=[draw, rounded corners, fill=orange!18,
    minimum width=24mm, minimum height=8mm, align=center]
  \tikzstyle{outputnode}=[draw, rounded corners, fill=green!18,
    minimum width=24mm, minimum height=8mm, align=center]

  \draw[->] (0,0) -- (5.1,0) node[below] {动作编号};
  \draw[->] (0,0) -- (0,1.35) node[left] {分数};

  \node[inputnode] at (0.9,1.22) {输入\\样本均值};
  \node[processnode] at (2.6,1.22) {处理\\加置信上界};
  \node[outputnode] at (4.25,1.22) {输出\\挑 UCB 最大};

  \draw[fill=blue!25] (0.55,0) rectangle (0.95,0.84);
  \draw[fill=blue!25] (1.55,0) rectangle (1.95,0.87);
  \draw[fill=blue!25] (2.55,0) rectangle (2.95,0.875);
  \draw[fill=blue!25] (3.55,0) rectangle (3.95,0.72);

  \draw[orange, very thick] (0.75,0.84) -- (0.75,1.0188);
  \draw[orange, very thick] (0.66,1.0188) -- (0.84,1.0188);

  \draw[orange, very thick] (1.75,0.87) -- (1.75,1.0249);
  \draw[orange, very thick] (1.66,1.0249) -- (1.84,1.0249);

  \draw[orange, very thick] (2.75,0.875) -- (2.75,1.0940);
  \draw[orange, very thick] (2.66,1.0940) -- (2.84,1.0940);

  \draw[orange, very thick] (3.75,0.72) -- (3.75,1.0297);
  \draw[orange, very thick] (3.66,1.0297) -- (3.84,1.0297);

  \node[below] at (0.75,0) {12};
  \node[below] at (1.75,0) {72};
  \node[below] at (2.75,0) {109};
  \node[below] at (3.75,0) {155};

  \draw[->, thick, green!60!black] (4.65,1.14) -- (2.86,1.10);
  \node[outputnode] at (4.5,1.14) {当前 UCB 最高\\优先试 109};
\end{tikzpicture}
```

图里的关键路径/要点：`action_id 109` 的蓝色柱子本来就不低，
而且试的次数还不算多，所以橙色“探索奖金”也比较可观，
最后总分冲到了最高。

### ④ DCVS 实际数值计算示例

假设到第 $t=11$ 轮为止，我们对 4 个动作已经看到了下面这些奖励：

- action_id 12：$0.84, 0.83, 0.85$
- action_id 72：$0.87, 0.86, 0.88, 0.87$
- action_id 109：$0.89, 0.86$
- action_id 155：$0.72$

先更新每个动作的样本均值和次数。

**action_id 12：**

$$
N_{12} = 3
$$

$$
\hat{\mu}_{12} = \frac{0.84 + 0.83 + 0.85}{3} = \frac{2.52}{3} = 0.84
$$

**action_id 72：**

$$
N_{72} = 4
$$

$$
\hat{\mu}_{72} = \frac{0.87 + 0.86 + 0.88 + 0.87}{4} = \frac{3.48}{4} = 0.87
$$

**action_id 109：**

$$
N_{109} = 2
$$

$$
\hat{\mu}_{109} = \frac{0.89 + 0.86}{2} = \frac{1.75}{2} = 0.875
$$

**action_id 155：**

$$
N_{155} = 1,
\qquad
\hat{\mu}_{155} = 0.72
$$

现在设 $c = 0.2$，先算公共项：

$$
\ln 11 \approx 2.3979
$$

接着逐个算探索奖金。

**action_id 12：**

$$
\sqrt{\frac{\ln 11}{3}} = \sqrt{\frac{2.3979}{3}} = \sqrt{0.7993} \approx 0.8940
$$

$$
0.2 \times 0.8940 = 0.1788
$$

$$
\text{UCB}_{12} = 0.84 + 0.1788 = 1.0188
$$

**action_id 72：**

$$
\sqrt{\frac{2.3979}{4}} = \sqrt{0.5995} \approx 0.7743
$$

$$
0.2 \times 0.7743 = 0.1549
$$

$$
\text{UCB}_{72} = 0.87 + 0.1549 = 1.0249
$$

**action_id 109：**

$$
\sqrt{\frac{2.3979}{2}} = \sqrt{1.19895} \approx 1.0950
$$

$$
0.2 \times 1.0950 = 0.2190
$$

$$
\text{UCB}_{109} = 0.875 + 0.2190 = 1.0940
$$

**action_id 155：**

$$
\sqrt{\frac{2.3979}{1}} = \sqrt{2.3979} \approx 1.5485
$$

$$
0.2 \times 1.5485 = 0.3097
$$

$$
\text{UCB}_{155} = 0.72 + 0.3097 = 1.0297
$$

最后比较 4 个 UCB 分数：

- $\text{UCB}_{12} = 1.0188$
- $\text{UCB}_{72} = 1.0249$
- $\text{UCB}_{109} = 1.0940$
- $\text{UCB}_{155} = 1.0297$

所以第 11 轮之后，UCB 会优先选择：

$$
a_{12} = \arg\max_a \text{UCB}_a = \text{action_id 109}
$$

这里很关键的一点是：UCB 没有盲目偏袒“最少试的动作”。
`action_id 155` 虽然只试过 1 次，但它均值太低，
所以最后还是被 `action_id 109` 压过去了。

## 4. 汤普森采样（Thompson Sampling）：像给每个动作都抽一张“今天可能有多强”的签

### ① 直觉类比

如果说 UCB 是“均值 + 探索奖金”的算分派，
那汤普森采样更像“每个动作先抽一次签，再让抽到最高签的那个上场”。

它不会直接比较固定分数，而是先承认：

- 我对每个动作的真实水平，其实都还不完全确定；
- 那不如先按当前认知，给每个动作随机抽一个“可能实力值”；
- 谁这次抽得最高，就选谁。

这种做法的妙处在于：

- 已经很强、而且证据充分的动作，经常会抽到高值；
- 潜力不明、但还没试够的动作，也偶尔能抽到高值，拿到探索机会。

### ② 正式定义

为了把概念讲得最简单，我们先把每次窗口结果压成“成功 / 失败”二元事件：

- 成功：FPS $\ge 59$，并且功耗 $\le 4500$ mW；
- 失败：其他情况。

这样每个动作的“成功概率”就可以记成 $\theta_a$。

在汤普森采样里，一个常见做法是给它一个 Beta 后验：

$$
\theta_a \sim \text{Beta}(\alpha_a, \beta_a)
$$

如果我们用最朴素的先验 $\text{Beta}(1,1)$，
那么看到若干次成功和失败之后，就有：

$$
\alpha_a = 1 + \text{成功次数},
\qquad
\beta_a = 1 + \text{失败次数}
$$

每一轮的决策规则是：

$$
\tilde{\theta}_a \sim \text{Beta}(\alpha_a, \beta_a),
\qquad
a_t = \arg\max_a \tilde{\theta}_a
$$

> **补充知识：Beta 分布（Beta Distribution）是什么**
>
> 先别把它想复杂。你可以把 Beta 分布理解成：
> “我对某个成功概率 $p$ 的主观信心，长什么样子”。
> 如果一开始完全没把握，就用 $\text{Beta}(1,1)$，它相当于在 $0$ 到 $1$ 之间比较平均。
>
> 如果像掷硬币一样，做了 $N$ 次试验，其中成功了 $K$ 次，
> 那么在 $\text{Beta}(1,1)$ 先验下，后验就会变成：
> $\text{Beta}(1+K, 1+N-K)$。
> 成功越多，曲线就越往“高成功率”那边偏；失败越多，就越往左偏。

### ③ 公式推导 + TikZ 图

这张图展示了什么：下面这张图用一条平的先验曲线和一条偏向右侧的后验曲线，
说明“观察到更多成功之后，我们会更相信这个动作本来就比较靠谱”。

```latex
\begin{tikzpicture}[>=Stealth]
  \tikzstyle{inputnode}=[draw, rounded corners, fill=blue!15,
    minimum width=24mm, minimum height=8mm, align=center]
  \tikzstyle{processnode}=[draw, rounded corners, fill=orange!18,
    minimum width=24mm, minimum height=8mm, align=center]
  \tikzstyle{outputnode}=[draw, rounded corners, fill=green!18,
    minimum width=24mm, minimum height=8mm, align=center]

  \draw[->] (0,0) -- (5.6,0) node[below] {成功概率 $p$};
  \draw[->] (0,0) -- (0,3.4) node[left] {可信程度};

  \draw[blue, thick] (0.5,1.1) .. controls (1.8,1.1) and (3.7,1.1) .. (5.0,1.1);
  \draw[orange, thick] (0.5,0.15) .. controls (2.2,0.5) and (3.3,3.1) .. (5.0,0.9);

  \node[inputnode] at (1.35,3.0) {先验\\Beta(1,1)};
  \node[processnode] at (3.0,3.0) {处理中间信息\\看到 4 成 1 败};
  \node[outputnode] at (4.65,3.0) {后验\\Beta(5,2)};

  \draw[->, thick] (1.85,2.8) -- node[above] {观测数据} (4.1,2.8);
\end{tikzpicture}
```

图里的关键路径/要点：先验是“还没试之前的看法”，
后验是“试过以后修正过的看法”。
汤普森采样每轮都会从这个后验里抽一个样本值出来做决策。

### ④ DCVS 实际数值计算示例

先做一个最简单的单动作例子。

假设 `action_id 72` 在 5 个窗口里：

- 成功 4 次；
- 失败 1 次。

如果先验是 $\text{Beta}(1,1)$，
那后验就是：

$$
\text{Beta}(1+4, 1+1) = \text{Beta}(5,2)
$$

它的后验均值可以顺手算一下：

$$
\mathbb{E}[\theta \mid \text{数据}] = \frac{\alpha}{\alpha + \beta} = \frac{5}{5+2} = \frac{5}{7}
$$

$$
\frac{5}{7} \approx 0.7143
$$

这不表示它下一次一定有 `71.43%` 的成功率，
而是说：结合目前证据，我们对它的成功概率判断，已经明显往右偏了。

现在把 4 个动作一起放进来：

- action_id 12：成功 2 次，失败 1 次，所以后验是 $\text{Beta}(3,2)$；
- action_id 72：成功 3 次，失败 1 次，所以后验是 $\text{Beta}(4,2)$；
- action_id 109：成功 2 次，失败 0 次，所以后验是 $\text{Beta}(3,1)$；
- action_id 155：成功 0 次，失败 1 次，所以后验是 $\text{Beta}(1,2)$。

接下来这一轮，汤普森采样会对每个动作各抽一次样本。
假设这次恰好抽到了：

- action_id 12：$0.58$
- action_id 72：$0.69$
- action_id 109：$0.83$
- action_id 155：$0.21$

因为 $0.83$ 最大，所以这一轮选：

$$
a_t = \text{action_id 109}
$$

下次再抽，结果可能会不同。
这正是它“自动平衡探索和利用”的地方：
高把握动作经常赢，但不是次次都垄断机会。

## 5. 用 4 个 action_id 走一遍完整 worked example

这一节我们把要求里的 4 个动作完整串起来，按“观测到的奖励序列 → 更新估计 →
选择下一个动作”的顺序走一遍。

### 观测到的奖励序列

假设前面已经记录到如下奖励：

- action_id 12：$0.84, 0.83, 0.85$
- action_id 72：$0.87, 0.86, 0.88, 0.87$
- action_id 109：$0.89, 0.86$
- action_id 155：$0.72$

### 更新估计

先更新次数：

- $N_{12} = 3$
- $N_{72} = 4$
- $N_{109} = 2$
- $N_{155} = 1$

再更新样本均值：

$$
\hat{\mu}_{12} = \frac{0.84 + 0.83 + 0.85}{3} = 0.84
$$

$$
\hat{\mu}_{72} = \frac{0.87 + 0.86 + 0.88 + 0.87}{4} = 0.87
$$

$$
\hat{\mu}_{109} = \frac{0.89 + 0.86}{2} = 0.875
$$

$$
\hat{\mu}_{155} = 0.72
$$

如果你走 UCB 路线，接下来会继续算探索奖金；
如果你走汤普森采样路线，接下来会把成功 / 失败次数更新成新的 Beta 后验。

### 选择下一个动作

沿用上一节的 UCB 结果：

- $\text{UCB}_{12} = 1.0188$
- $\text{UCB}_{72} = 1.0249$
- $\text{UCB}_{109} = 1.0940$
- $\text{UCB}_{155} = 1.0297$

因此，下一个动作会选：

$$
\text{next action} = \text{action_id 109}
$$

如果换成汤普森采样，
只要这一轮从各自后验里抽出来的样本值里，109 仍然最大，
结果也会落到 `action_id 109`。

这个 worked example 想传达的核心只有一句话：

- Bandit 不是一次算出永远正确的动作；
- 它是每一轮都用最新观测，把“下一次更值得试谁”往前推一步。

## 6. plain bandit 的局限：同一个动作，不会在所有场景里都一样好

### ① 直觉类比

同一把伞，在暴雨天和大太阳天的价值，显然不一样。
同一个 DCVS 动作，在团战、高负载、发热明显的时候，
和在菜单、低负载、温度很低的时候，表现也不会一样。

### ② 正式定义

plain bandit 默认有一个很强的假设：

$$
\mu_a \text{ 是固定的}
$$

也就是它假设：动作 $a$ 的平均回报，不随上下文变化。

但 DCVS 的实际情况更像：

$$
\mu(a, x_t) \text{ 会随着上下文 } x_t \text{ 变化}
$$

这里的上下文 $x_t$ 可能包括：

- 当前 GPU 利用率；
- 当前游戏场景；
- 当前温度与热状态；
- 前 1 到 3 个窗口的历史统计；
- 上一个动作和上一个奖励。

### ③ 公式推导 + TikZ 图

这张图展示了什么：同一个 `action_id 72`，放在两个完全不同的上下文里，
拿到的奖励可能差很多。也正因为如此，plain bandit 很快就不够用了。

```latex
\begin{tikzpicture}[>=Stealth, node distance=12mm and 10mm]
  \tikzstyle{inputnode}=[draw, rounded corners, fill=blue!15,
    minimum width=32mm, minimum height=9mm, align=center]
  \tikzstyle{processnode}=[draw, rounded corners, fill=orange!18,
    minimum width=34mm, minimum height=9mm, align=center]
  \tikzstyle{outputnode}=[draw, rounded corners, fill=green!18,
    minimum width=34mm, minimum height=9mm, align=center]

  \node[inputnode] (action) {输入\\同一个动作\\action\_id 72};
  \node[processnode, below left=of action] (heavy) {处理\\团战高负载\\温度持续升高};
  \node[processnode, below right=of action] (light) {处理\\菜单低负载\\温度较低};
  \node[outputnode, below=of heavy] (heavyout) {输出\\FPS 稳 60\\奖励约 0.87};
  \node[outputnode, below=of light] (lightout) {输出\\FPS 也稳\\但功耗偏高\\奖励可能不如低频动作};

  \draw[->, thick] (action) -- node[left] {放进高负载上下文} (heavy);
  \draw[->, thick] (action) -- node[right] {放进低负载上下文} (light);
  \draw[->, thick] (heavy) -- node[right] {得到结果} (heavyout);
  \draw[->, thick] (light) -- node[left] {得到结果} (lightout);
\end{tikzpicture}
```

图里的关键路径/要点：不是动作本身“永远好”或者“永远坏”，
而是动作和上下文要配套看。

### ④ DCVS 实际数值计算示例

还是拿 `action_id 72` 来看。

**场景 A：团战，高负载。**

- GPU 频率：`545 MHz`
- FPS：`60`
- 功耗：`4200 mW`

前面已经算过：

$$
r_A = 0.86875
$$

**场景 B：菜单，低负载。**

这时如果你还用 `action_id 72`，
假设结果变成：

- GPU 频率：`545 MHz`
- FPS：`60`
- 功耗：`5200 mW`

那奖励会变成：

$$
r_{\text{base},B} = \frac{60}{60} - 0.25 \cdot \frac{5200}{8000}
$$

先算功耗项：

$$
\frac{5200}{8000} = 0.65
$$

$$
0.25 \cdot 0.65 = 0.1625
$$

所以：

$$
r_B = 1 - 0.1625 = 0.8375
$$

也就是说，同一个动作在两个上下文里的奖励差了：

$$
0.86875 - 0.8375 = 0.03125
$$

别小看这个差值。在线调很多轮以后，它会持续积累，
最后把动作排序完全改写。

这也是为什么 plain bandit 只是第一步。
一旦你意识到“上下文变了，动作好坏也会变”，
下一个更自然的框架就是**上下文老虎机（Contextual Bandit）**。

## 7. 和项目源文档连起来看：为什么有人推 MDP，有人推 contextual bandit

到这里，你应该能理解三份源材料为什么会出现不同判断了。

- 如果你把问题看成“每个小窗口先根据当前情况选一个动作，立刻看奖励”，
  那它很像 bandit，运行时文档自然会偏向汤普森采样，
  甚至明确提到安全动作子集（safe action subset）。
- 如果你把热积累、governor 惯性、前几个窗口的影响都看得更重，
  那 plain bandit 就过于简化，建模上会更靠近 MDP。
- 如果你注意到当前离线数据覆盖并不充分，
  比如现有 `offline_test_iter5` 样本里明显偏向单一 `action_id`，
  那你也会知道：不管是 UCB 还是汤普森采样，在线阶段都得格外重视安全约束。

所以比较稳的工程结论通常不是“Bandit 永远最好”或者“MDP 永远最好”，
而是：

- plain bandit 适合拿来搭最小在线决策闭环；
- 上下文老虎机适合把当前场景差异纳进来；
- 更强的时序依赖，再往 MDP 和完整 RL 走。

## 关键收获

- 多臂老虎机（Multi-Armed Bandit）最适合帮助你建立“有限试错预算下怎么在线选动作”的直觉。
- 累积遗憾（Cumulative Regret）衡量的不是你这轮有没有赢，而是你离最优动作还差了多少。
- 样本均值和大数定律告诉我们：动作试得越多，均值估计通常越稳。
- UCB 的核心就是“经验均值 + 不确定性奖金”，它会优先给“看起来不错、但还没试够”的动作机会。
- 汤普森采样（Thompson Sampling）会给每个动作维护一个不确定性分布，再通过随机抽样完成探索。
- Beta 分布（Beta Distribution）之所以常出现，是因为它很适合描述“成功概率到底有多大”这件事。
- plain bandit 的大前提是“每个动作的平均好坏基本固定”，而这在 DCVS 里通常不成立。
- 当 GPU 负载、温度、游戏场景持续变化时，就该从 plain bandit 迈向上下文老虎机（Contextual Bandit）。
