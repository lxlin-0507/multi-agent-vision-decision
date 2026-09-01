# 多agent视觉感知与可解释决策

> 一个面向图像分析任务的多 Agent 决策工作台：将目标检测、场景分析、质量复核、评分决策与可追溯报告整合为一次可审计的工作流。

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-6.x-F97316)
![LangGraph](https://img.shields.io/badge/LangGraph-Workflow-1C3C3C)

## 项目亮点

- **多 Agent 编排**：视觉感知、场景分析、质量复核、决策评分和报告持久化以 LangGraph 编排。
- **可解释决策**：输出检测数量、置信度、图像质量、评分分解、人工复核原因和完整审计链路。
- **视觉工作台**：提供证据序列、检测结果主画布、结构化报告、关键证据裁剪和报告导出。
- **可用交互**：主画布支持拖拽、滚轮缩放、按钮缩放、复位和快捷键控制。
- **智能追问**：基于本次分析上下文接入 DeepSeek，支持流式多轮问答。
- **结果可恢复**：在同一浏览器会话内切换工作台与智能追问，不会重复运行分析或丢失结果。

## 工作流

```text
输入图像
  → 视觉感知（YOLO）
  → Supervisor 路由
      ├─ 场景分析
      └─ 质量复核
  → 决策评分
  → 结构化报告 + 审计产物
  → 工作台展示 / 智能追问
```

## 快速开始

### 1. 创建环境

推荐使用 Python 3.12：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

如需使用“智能追问”或 LLM 报告生成，在 `.env` 中填入 `DEEPSEEK_API_KEY`。不要将 `.env` 提交到 Git。

### 3. 启动应用

```bash
APP_PORT=7861 GRADIO_SERVER_NAME=127.0.0.1 python app.py
```

打开 <http://127.0.0.1:7861>。

首次真实推理需要可用的 YOLO 权重。默认 `yolov8n.pt` 可由 Ultralytics 下载；生产环境建议通过挂载卷或模型仓库提供权重文件。

## 测试

```bash
python -m pytest -q
```

测试覆盖评分规则、工作流路由、数据校验、流式追问和工作台状态恢复。

## 目录结构

```text
.
├── app.py                       # Gradio 工作台入口
├── enhanced_auditable_agent.py  # LangGraph 工作流与报告持久化
├── chat_agent.py                # DeepSeek 流式追问
├── agents/                      # 视觉、场景、质量、评分 Agent
├── tests/                       # 自动化测试
├── scoring_rules.yaml           # 可解释评分规则
├── .env.example                 # 环境变量模板，不含密钥
├── Dockerfile                   # 容器化部署入口
└── render.yaml                  # Render 部署声明
```

## 部署

项目提供 Dockerfile，可部署至 Render、Railway 或任意 Docker 平台。以 Render 为例：

1. 将仓库推送到 GitHub。
2. 在 Render 中使用仓库根目录的 `render.yaml` 创建 Web Service。
3. 在服务环境变量中填写 `DEEPSEEK_API_KEY`，不要写入仓库。
4. 为模型权重和 `reports/` 配置持久化存储（如需要保留产物）。

## 技术栈

Python · Gradio · LangGraph · Ultralytics YOLO · Pydantic · DeepSeek API

## 安全说明

- `.env`、日志、缓存、报告与模型权重均已由 `.gitignore` 排除。
- 推送前请确认历史提交、截图和文档中不含 API Key；如密钥曾泄露，请先在供应商控制台轮换。
