# Agent Instructions

## Project Overview

Generate a beginner-to-advanced learning curriculum for reinforcement learning applied to GPU DCVS tuning. All generated learning documents must live under the `learn/` subfolder in this workspace.

## Scope Boundaries

- Only create or update files inside `learn/`.
- Do not modify the three source analysis documents in the workspace root.
- Do not modify files under `.ralphy/` while executing PRD tasks.
- Do not create code, scripts, or datasets outside the learning materials requested by the task list.

## Required Source Materials

You must synthesize the learning materials from these sources:

- Workspace source documents:
	- `RL_Algorithm_Selection_for_GPU_DCVS_Tuning_20260401.md`
	- `RL_DCVS_AI_Integrated_Decision_2026-04-01.md`
	- `RL_DCVS_Runtime_Adaptive_Independent_Decision_GPT-5.4_2026-04-01_135421.md`
- External context:
	- `D:\D_Code_2\QCOM-RL\collect_qgtf_dataset.py`
	- `D:\D_Code_2\QCOM-RL\offline_test_iter5`

## 写作语言与风格

- **全部用中文书写**，语言要通俗易懂、口语化，像一个耐心的导师在面对面讲解。
- 避免直接翻译英文论文的生硬学术腔（例如不要写"我们考虑一个马尔可夫决策过程"，而要写"想象你是一个调参工程师，每隔几秒就要做一次决定……"）。
- 首次出现的专业术语必须同时给出英文原名，格式为：中文名（English Term）。后续可以只用中文。
- 图表说明文字也必须用中文。

## 目标读者画像

- **线性代数只有入门水平**：知道矩阵是什么、能做简单矩阵乘法，但不熟悉特征值分解、逆矩阵的几何意义等。
- **概率论基础薄弱**：知道概率、期望、条件概率的定义，但不熟悉贝叶斯推断、概率分布族等。
- **微积分有基础**：知道导数和积分的基本概念，但不会推复杂的多元链式法则。
- 对于任何超出上述水平的数学工具（如矩阵求逆、概率分布采样、梯度下降推导、KL散度等），必须插入一个"补充知识"框（用Markdown blockquote `>` 格式），用1-3段话把该前置知识从零讲清楚，**不要假设读者已经掌握**。

## 数学公式与可视化要求

- 所有数学公式使用 LaTeX 格式：行内用 `$...$`，独立公式块用 `$$...$$`。
- **对于复杂的计算流程、算法流程、公式推导过程和数值计算步骤，必须用 LaTeX + TikZ 绘制流程图或示意图**，以 ```` ```latex ```` 代码块嵌入 Markdown 文件中，让读者可以用任意 LaTeX 编译器渲染查看。
- TikZ 图的要求：
  - 每个节点和箭头都要有中文标注。
  - 用不同颜色区分输入/处理/输出节点。
  - 图的复杂度要适中：一张图不超过15个节点，超过的拆成多张。
  - 在图之前用一段文字说明"这张图展示了什么"，在图之后用一段文字解读"图中的关键路径/要点"。
- 数值计算示例要写出**每一步的中间结果**，不要跳步，让读者能用笔跟着算。

## 概念讲解四步法

对于每一个新概念，必须遵循以下顺序：
1. **直觉类比**：用日常生活或工程场景的例子解释这个概念在干什么（例如把Q值类比成"经验评分表"）。
2. **正式定义**：给出数学或技术上的严格定义。
3. **公式推导 + TikZ 图**：如果涉及公式推导，画一张 TikZ 流程图展示推导的逻辑链路；如果是算法流程，画算法流程图。
4. **DCVS 实际数值计算示例**：用项目中的真实数据范围（GPU频率282-710 MHz，FPS目标59-60，功耗0-8000 mW，192个离散动作等）做一个完整的带数字的计算过程。

## Writing Requirements

- Every document must be self-contained.
- Every document must start with a `前置阅读` section listing which previous documents should be read first.
- Every document must end with a `关键收获` summary section.
- Every document must include clear mathematical formulas using LaTeX.
- Every formula-heavy section must include at least one worked numerical example with full intermediate steps.
- Numerical examples should use realistic DCVS values from the project context, such as FPS target 59-60, GPU frequency 282-710 MHz, action space size 192, and power-related metrics from the collected dataset.

## Content Priorities

- Help a reader with weak reinforcement learning background build understanding from first principles.
- When introducing matrix operations, probability distributions, gradient descent, or any math tool beyond basic calculus, always provide an inline "补充知识" box explaining it from scratch.
- Explain why different source documents disagree on the best method and what assumptions drive each recommendation.
- Highlight the relationship between problem formulation and algorithm choice: bandit vs MDP, online vs offline RL, discrete vs continuous action spaces, safety constraints, and data coverage limitations.
- Make the CQL, IQL, contextual bandit, reward design, state design, and action design sections especially rigorous.

## Quality Bar

- Avoid vague summaries.
- Prefer concrete state, action, reward, and transition examples from the DCVS problem.
- If a concept is abstract, add a small table or step-by-step numeric example.
- Keep the learning order progressive from easy to hard.
- Every document with algorithms must contain at least one TikZ diagram (algorithm flowchart, computation graph, or concept map).
