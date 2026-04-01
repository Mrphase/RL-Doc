$ErrorActionPreference = 'Stop'

$doc = Join-Path $PSScriptRoot '..\04_Multi_Armed_Bandits.md'

if (-not (Test-Path $doc)) {
    throw "缺少目标文档: $doc"
}

$lines = Get-Content $doc -Encoding UTF8

for ($index = 0; $index -lt $lines.Count; $index++) {
    $line = $lines[$index]
    $lineNo = $index + 1

    if ($line -match "`t") {
        throw "第 $lineNo 行包含 Tab 字符，不符合 Markdown 排版约束"
    }

    if ($line -match '\s+$') {
        throw "第 $lineNo 行存在行尾空白"
    }

    if ($line.Length -gt 220) {
        throw "第 $lineNo 行长度超过 220 个字符，当前长度为 $($line.Length)"
    }
}

$raw = Get-Content $doc -Raw -Encoding UTF8
if (-not $raw.EndsWith("`n")) {
    throw '文件末尾缺少换行'
}

if ($raw.Contains('assets/tikz-preview')) {
    throw '正文不应再依赖 tikz-preview 静态图资源'
}

$appendixHeading = '## 附录：TikZ源码（可选）'
$appendixIndex = $raw.IndexOf($appendixHeading)
$latexMatches = [regex]::Matches($raw, '```latex')
if ($appendixIndex -lt 0) {
    if ($latexMatches.Count -gt 0) {
        throw '正文不应包含 LaTeX 图代码块；如需保留 TikZ，请放到附录里'
    }
}
else {
    foreach ($match in $latexMatches) {
        if ($match.Index -lt $appendixIndex) {
            throw '检测到附录标题之前出现 LaTeX 图代码块，正文图必须改用 Mermaid'
        }
    }
}

$mermaidBlocks = ([regex]::Matches($raw, '```mermaid')).Count
if ($mermaidBlocks -lt 3) {
    throw "Mermaid 图数量不足，当前为 $mermaidBlocks，期望至少 3 个"
}

Write-Host 'PASS: Markdown 本地 lint 通过'
