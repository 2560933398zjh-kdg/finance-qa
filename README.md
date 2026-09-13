# 家庭理财 AI 智能问答系统

基于本地理财知识库（PDF 论文 + 文本资料）的 **家庭理财 AI 问答系统**。支持 BM25 / Elasticsearch 双检索，使用通义千问大模型生成回答，通过 Gradio 提供 Web 界面。

## 项目简介

系统将理财学术论文（PDF）与理财知识文档解析后构建为本地知识库，用户提问时先检索相关知识片段，再由 Qwen 大模型结合检索结果生成回答。适用于"家庭理财规划""基金定投""保险配置"等主题的智能咨询。

## 技术栈

- Python + Gradio（Web 界面）
- pypdf（PDF 解析）
- BM25 检索 + Elasticsearch（可选，文档量大时启用）
- qwen_agent / dashscope（通义千问大模型生成）

## 目录结构

```
finance-qa/
├── finance_advisor.py       # 主程序：理财问答系统（Gradio）
├── finance_2.py             # 备用版本（含 ES 可用性检测）
├── requirements.txt         # 依赖清单
├── docs/                    # 理财学术论文 PDF（5 篇，知识库来源）
│   ├── 互联网金融环境下大学生理财方式的探讨_吴奇.pdf
│   ├── 城市家庭的经济条件、理财意识和投资借贷行为_廖理.pdf
│   ├── 我国家庭理财规划浅议_欧阳红兵.pdf
│   ├── 投资者需提升理财意识和能力_许予朋.pdf
│   └── 浅谈银行理财产品的营销技巧和体会_孙章毅.pdf
├── finance_docs/            # 理财知识文本（5 个主题）
│   ├── 个人所得税专项附加扣除.txt
│   ├── 基金定投策略.txt
│   ├── 家庭保险配置指南.txt
│   ├── 房贷LPR与还款策略.txt
│   └── 股票投资基础术语.txt
└── workspace/tools/         # qwen_agent 工具工作区
```

## 运行方式

```bash
pip install -r requirements.txt
python finance_advisor.py
```

## 配置说明

- 通过环境变量 `DASHSCOPE_API_KEY` 配置密钥：复制 `.env.example` 为 `.env` 并填入你的 DashScope API Key（`.env` 已被 .gitignore 忽略，不会提交）。运行前在 PowerShell 执行：`$env:DASHSCOPE_API_KEY = (Get-Content .env | Where-Object {$_ -like 'DASHSCOPE*'} | ForEach-Object {($_ -split '=')[1]})`
- 模型默认 `qwen-turbo`，可在代码中改为 `qwen-plus` / `qwen-max`。
- `ES_CONFIG` 为可选 Elasticsearch 配置；文档数量不大时系统默认使用 BM25 本地检索，无需启动 ES。
