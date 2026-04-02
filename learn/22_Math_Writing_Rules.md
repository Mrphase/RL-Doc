<!-- markdownlint-disable MD041 MD013 -->
# 22_Math_Writing_Rules：`learn/` 文档里的数学公式书写规则

这份规则文档只解决一个很具体的问题：**避免在 Markdown 数学公式里重复写出会导致渲染报错的 LaTeX 写法。**

这次踩到的典型报错是：

> `'_' allowed only in math mode`

它对应的高频场景是：**把带下划线的标识符放进 `\text{...}` 里，但下划线没有按数学模式安全处理。**

---

## 1. 先记结论

在 `learn/` 目录下写公式时，遇到这类名字：

- `first_step_down`
- `penalty_down`
- `penalty_up`
- `strict_frame`
- `fps_component`
- `gpu_freq_mhz`
- `power_mw`

统一按下面的规则写：

1. **只要是公式里的“变量名 / 字段名 / 参数名 / 组件名”，优先用数学模式写。**
2. **如果名字里带下划线，不要直接写成普通文本。**
3. **需要直立体标识符时，优先写成 `\mathrm{...}`，并把下划线转义成 `\_`。**
4. **` \text{...} ` 更适合写自然语言短语，不适合反复包裹这类带下划线的技术标识符。**

---

## 2. 这次错误到底是怎么来的

下面这种写法就是这次问题的来源之一：

```latex
$$
a_t = (\text{first_step_down}=10,\ \text{penalty_down}=90,\ \text{penalty_up}=85,\ \text{strict_frame}=0)
$$
```

这里的问题不是“公式里不能出现字段名”，而是：

- 这些名字本质上是公式里的标识符；
- 它们包含下划线；
- 直接放进 `\text{...}` 后，渲染器很容易把它当成不安全或不规范的写法。

更稳妥的写法是：

```latex
$$
a_t = (\mathrm{first\_step\_down}=10,\ \mathrm{penalty\_down}=90,\ \mathrm{penalty\_up}=85,\ \mathrm{strict\_frame}=0)
$$
```

---

## 3. 推荐写法与禁用写法

### 3.1 动作、状态字段、奖励组件

推荐：

```latex
\mathrm{first\_step\_down}
\mathrm{penalty\_down}
\mathrm{penalty\_up}
\mathrm{strict\_frame}
\mathrm{fps\_component}
\mathrm{freq\_component}
\mathrm{power\_component}
\mathrm{gpu\_freq\_mhz}
\mathrm{power\_mw}
```

避免：

```latex
\text{first_step_down}
\text{penalty_down}
\text{gpu_freq_mhz}
```

### 3.2 动作元组

推荐：

```latex
$$
a_t = (\mathrm{first\_step\_down}=10,\ \mathrm{penalty\_down}=90,\ \mathrm{penalty\_up}=85,\ \mathrm{strict\_frame}=0)
$$
```

避免：

```latex
$$
a_t = (\text{first_step_down}=10,\ \text{penalty_down}=90,\ \text{penalty_up}=85,\ \text{strict_frame}=0)
$$
```

### 3.3 奖励函数里的组件名

推荐：

```latex
$$
\mathrm{reward} = \mathrm{fps\_component} + \mathrm{freq\_component} + \mathrm{power\_component}
$$
```

避免：

```latex
$$
\text{reward} = \text{fps_component} + \text{freq_component} + \text{power_component}
$$
```

---

## 4. 什么时候可以继续用 `\text{...}`

` \text{...} ` 不是不能用，而是要用在合适的位置。

更适合它的场景：

- 在公式里插一句自然语言说明；
- 写 `if`, `otherwise` 一类短语；
- 写不带下划线的普通说明词；
- 分段函数里给条件做简短文本注释。

例如下面这种通常没问题：

```latex
\text{if } fps < 59
```

但下面这种字段名写法，不建议继续扩散：

```latex
\text{strict_frame}
```

---

## 5. 写公式前的快速自检清单

以后在 `learn/` 新增或修改公式，提交前至少扫一遍下面 5 件事：

- [ ] 公式里的字段名、参数名、奖励组件名，是不是被当成了数学标识符来写？
- [ ] 只要名字里有下划线，是不是已经写成了 `\_`？
- [ ] 这类技术标识符是不是优先用了 `\mathrm{...}`，而不是 `\text{...}`？
- [ ] 动作元组、状态字段、奖励函数三类高频公式，写法是不是和已有文档保持一致？
- [ ] 如果复制旧公式，是否顺手把旧的 `\text{字段名}` 写法也一起修正了？

---

## 6. 一条可直接执行的仓库规则

以后在 `learn/` 目录中，凡是公式里的**带下划线技术标识符**，统一遵守下面这条规则：

> **用 `\mathrm{...}` 包裹，并把每个下划线写成 `\_`；不要把这类名字直接写进 `\text{...}`。**

如果后面继续新增 CQL、IQL、A2C、PPO、SAC、DT 等文档中的公式，都按这条规则执行，这次这类渲染错误基本就不会再重复出现。
