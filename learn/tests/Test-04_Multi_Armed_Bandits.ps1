$ErrorActionPreference = 'Stop'

$doc = Join-Path $PSScriptRoot '..\\04_Multi_Armed_Bandits.md'

function Assert-Contains {
    param(
        [string]$Content,
        [string]$Needle,
        [string]$Message
    )

    if (-not $Content.Contains($Needle)) {
        throw $Message
    }
}

if (-not (Test-Path $doc)) {
    throw "缺少目标文档: $doc"
}

$content = Get-Content $doc -Raw -Encoding UTF8

Assert-Contains $content '## 前置阅读' '缺少“前置阅读”部分'
Assert-Contains $content '多臂老虎机（Multi-Armed Bandit）' '缺少多臂老虎机中英术语'
Assert-Contains $content '赌场' '缺少赌场老虎机类比'
Assert-Contains $content '192 个 DCVS 参数组合' '缺少 192 个 DCVS 参数组合映射'
Assert-Contains $content '累积遗憾（Cumulative Regret）' '缺少累积遗憾术语'
Assert-Contains $content '样本均值' '缺少样本均值说明'
Assert-Contains $content '大数定律' '缺少大数定律说明'
Assert-Contains $content 'a_t = \arg\max_a \left[\hat{\mu}_a + c\sqrt{\frac{\ln t}{N_a}}\right]' '缺少 UCB 公式'
Assert-Contains $content 'Thompson Sampling' '缺少 Thompson Sampling'
Assert-Contains $content 'Beta 分布（Beta Distribution）' '缺少 Beta 分布术语'
Assert-Contains $content '上下文老虎机（Contextual Bandit）' '缺少上下文老虎机引出'
Assert-Contains $content 'action_id 12' '缺少 worked example 的 action_id 12'
Assert-Contains $content 'action_id 72' '缺少 worked example 的 action_id 72'
Assert-Contains $content 'action_id 109' '缺少 worked example 的 action_id 109'
Assert-Contains $content 'action_id 155' '缺少 worked example 的 action_id 155'
Assert-Contains $content '282-710 MHz' '缺少 GPU 频率范围'
Assert-Contains $content '59-60' '缺少 FPS 目标范围'
Assert-Contains $content '0-8000 mW' '缺少功耗范围'
Assert-Contains $content '6×4×4×2' '缺少动作空间分解'
Assert-Contains $content '## 关键收获' '缺少“关键收获”部分'

$latexBlocks = ([regex]::Matches($content, '```latex')).Count
if ($latexBlocks -lt 3) {
    throw "LaTeX 代码块数量不足，当前为 $latexBlocks，期望至少 3 个"
}

$tikzBlocks = ([regex]::Matches($content, '\\begin\{tikzpicture\}')).Count
if ($tikzBlocks -lt 3) {
    throw "TikZ 图数量不足，当前为 $tikzBlocks，期望至少 3 个"
}

$formulaBlocks = ([regex]::Matches($content, '\$\$')).Count
if ($formulaBlocks -lt 10) {
    throw "公式块标记数量不足，当前为 $formulaBlocks，期望至少 10 个"
}

$knowledgeBoxes = ([regex]::Matches($content, '^> \*\*补充知识', [System.Text.RegularExpressions.RegexOptions]::Multiline)).Count
if ($knowledgeBoxes -lt 2) {
    throw "补充知识框数量不足，当前为 $knowledgeBoxes，期望至少 2 个"
}

Assert-Contains $content '观测到的奖励序列' '缺少 worked example 的奖励序列说明'
Assert-Contains $content '更新估计' '缺少 worked example 的更新估计说明'
Assert-Contains $content '选择下一个动作' '缺少 worked example 的选择动作说明'
Assert-Contains $content 'Beta(1,1)' '缺少 Beta 先验示例'
Assert-Contains $content 'Beta(5,2)' '缺少 Beta 后验示例'
Assert-Contains $content 'safe action subset' '缺少安全动作子集说明'

Write-Host 'PASS: learn/04_Multi_Armed_Bandits.md 内容校验通过'
