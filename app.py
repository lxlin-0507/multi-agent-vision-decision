"""
多agent视觉感知与可解释决策界面。
"""
import html
import os
from pathlib import Path
from typing import Any, Dict

import config
import gradio as gr
from PIL import Image, UnidentifiedImageError

from exceptions import AgentBaseError
from enhanced_auditable_agent import run_pipeline
from chat_agent import get_or_create_chat, clear_chat
from logger import get_logger

_log = get_logger("gradio_ui")

APP_NAME = config.APP_NAME
APP_THEME = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
)

APP_CSS = """
:root {
    --atlas-canvas: #eef2f5;
    --atlas-paper: #ffffff;
    --atlas-ink: #17212b;
    --atlas-muted: #687583;
    --atlas-line: #dce3e8;
    --atlas-line-soft: #e7edf1;
    --atlas-coral: #3b82f6;
    --atlas-coral-soft: #dbeafe;
    --atlas-wash: #f7f9fb;
    --atlas-shadow: 0 10px 24px rgba(21, 33, 43, 0.08);
    --atlas-body: "Noto Sans SC", "Source Han Sans SC", "PingFang SC",
        "Microsoft YaHei", sans-serif;
    --atlas-latin: Inter, "IBM Plex Sans", "Noto Sans SC", "PingFang SC",
        sans-serif;
    --atlas-mono: "IBM Plex Mono", "SFMono-Regular", "Roboto Mono",
        "Noto Sans Mono", monospace;
}

html,
body {
    min-width: 320px;
    margin: 0;
    background: var(--atlas-canvas);
}

.gradio-container {
    width: 100% !important;
    max-width: none !important;
    min-height: 100vh;
    padding: 0 !important;
    color: var(--atlas-ink) !important;
    font-family: var(--atlas-body) !important;
    background: var(--atlas-canvas) !important;
}

.gradio-container > .main {
    max-width: none !important;
    padding: 0 !important;
}

#atlas-topbar {
    position: sticky;
    z-index: 30;
    top: 0;
    display: grid !important;
    grid-template-columns: minmax(0, 1fr) auto auto;
    gap: 18px !important;
    align-items: center;
    min-height: 56px;
    margin: 0 !important;
    padding: 0 20px !important;
    border-bottom: 1px solid #263746 !important;
    background: #15212b !important;
}

#atlas-topbar > * {
    min-width: 0;
}

.atlas-brand {
    display: flex;
    align-items: center;
    gap: 11px;
    min-width: 0;
}

.atlas-mark {
    position: relative;
    width: 25px;
    height: 25px;
    flex: 0 0 25px;
}

.atlas-mark::before,
.atlas-mark::after {
    position: absolute;
    inset: 4px;
    border: 1px solid #dceaff;
    content: "";
}

.atlas-mark::before {
    border-radius: 50% 50% 50% 2px;
    transform: rotate(45deg);
}

.atlas-mark::after {
    inset: 10px;
    border-radius: 50%;
    background: var(--atlas-coral);
}

.brand-copy {
    min-width: 0;
}

.brand-title {
    overflow: hidden;
    margin: 0;
    color: #f8fbff;
    font: 600 14px/1.2 var(--atlas-body);
    text-overflow: ellipsis;
    white-space: nowrap;
}

.brand-caption {
    margin: 3px 0 0;
    color: #a8b5c3;
    font: 500 9px/1 var(--atlas-latin);
    letter-spacing: 0.13em;
    text-transform: uppercase;
}

#export-report {
    min-width: 118px !important;
    min-height: 34px !important;
    border: 1px solid var(--atlas-coral) !important;
    border-radius: 5px !important;
    color: #ffffff !important;
    font-size: 12px !important;
    font-weight: 650 !important;
    background: var(--atlas-coral) !important;
    box-shadow: none !important;
}

#export-report:hover:not([disabled]) {
    border-color: #2563eb !important;
    background: #2563eb !important;
}

#export-report[disabled] {
    opacity: 0.46 !important;
}

.more-action {
    display: grid;
    width: 32px;
    height: 32px;
    place-items: center;
    border: 0;
    color: #b6c4d0;
    background: transparent;
    cursor: not-allowed;
}

.more-action svg {
    width: 18px;
    height: 18px;
}

#mode-tabs > .tab-nav {
    position: sticky;
    z-index: 20;
    top: 56px;
    gap: 24px;
    min-height: 42px;
    padding: 0 20px !important;
    border-bottom: 1px solid var(--atlas-line) !important;
    background: #f8fafc !important;
}

#mode-tabs > .tab-nav button,
#report-tabs > .tab-nav button {
    min-width: auto !important;
    padding: 11px 1px 9px !important;
    border: 0 !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    color: var(--atlas-muted) !important;
    font-size: 12px !important;
    font-weight: 550 !important;
    background: transparent !important;
    box-shadow: none !important;
}

#mode-tabs > .tab-nav button.selected,
#report-tabs > .tab-nav button.selected {
    border-bottom-color: var(--atlas-coral) !important;
    color: var(--atlas-ink) !important;
}

#atlas-workspace {
    display: grid !important;
    grid-template-columns: 220px minmax(0, 1fr) 320px;
    gap: 0 !important;
    min-height: calc(100vh - 98px);
    margin: 0 !important;
}

#atlas-workspace > * {
    min-width: 0 !important;
}

.evidence-rail,
.report-rail {
    padding: 20px 17px 24px !important;
    background: var(--atlas-canvas) !important;
}

.evidence-rail {
    border-right: 1px solid var(--atlas-line) !important;
}

.report-rail {
    overflow-y: auto;
    max-height: calc(100vh - 98px);
    border-left: 1px solid var(--atlas-line) !important;
}

.image-stage {
    min-height: calc(100vh - 98px);
    padding: 0 !important;
    background: #dfe7ed !important;
}

.rail-heading,
.canvas-heading {
    margin: 0;
    color: var(--atlas-ink);
    font-size: 14px;
    font-weight: 620;
    letter-spacing: -0.01em;
}

.rail-intro {
    margin: 6px 0 18px;
    color: var(--atlas-muted);
    font-size: 11px;
    line-height: 1.55;
}

.evidence-entry {
    margin-top: 15px;
}

.evidence-entry-head {
    display: flex;
    justify-content: space-between;
    gap: 8px;
    align-items: baseline;
    margin-bottom: 7px;
}

.evidence-name {
    color: var(--atlas-ink);
    font-size: 11px;
    font-weight: 600;
}

.evidence-state {
    color: var(--atlas-muted);
    font: 500 9px/1 var(--atlas-mono);
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.evidence-image {
    overflow: hidden;
    border: 1px solid var(--atlas-line) !important;
    border-radius: 6px !important;
    background: var(--atlas-wash) !important;
    box-shadow: none !important;
}

.evidence-image.is-current {
    border: 2px solid var(--atlas-coral) !important;
}

.evidence-image .image-container,
.evidence-image img {
    border-radius: 0 !important;
}

#run-button,
#batch-run-button,
#send-button,
.secondary-action {
    min-height: 38px !important;
    border: 1px solid var(--atlas-line) !important;
    border-radius: 5px !important;
    color: var(--atlas-ink) !important;
    font-size: 12px !important;
    font-weight: 620 !important;
    background: transparent !important;
    box-shadow: none !important;
}

#run-button {
    margin-top: 17px;
    border-color: var(--atlas-coral) !important;
    color: #ffffff !important;
    background: var(--atlas-coral) !important;
}

#run-button:hover,
#batch-run-button:hover,
#send-button:hover,
.secondary-action:hover {
    background: #2563eb !important;
}

.support-note p {
    margin: 9px 0 0 !important;
    color: var(--atlas-muted) !important;
    font-size: 10px !important;
    line-height: 1.55 !important;
}

.canvas-topline {
    display: flex;
    justify-content: space-between;
    gap: 14px;
    align-items: center;
    min-height: 48px;
    padding: 0 16px;
    border-bottom: 1px solid var(--atlas-line);
    background: #f8fafc;
}

.canvas-heading-wrap {
    display: flex;
    gap: 9px;
    align-items: center;
}

.canvas-status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--atlas-coral);
}

.viewer-toolbar {
    display: flex;
    gap: 3px;
    align-items: center;
    padding: 3px;
    border: 1px solid rgba(72, 91, 107, 0.38);
    background: rgba(21, 33, 43, 0.94);
    box-shadow: var(--atlas-shadow);
}

.viewer-tool {
    display: grid;
    width: 27px;
    height: 27px;
    padding: 0;
    place-items: center;
    border: 1px solid transparent;
    border-radius: 1px;
    color: #c3d0dc;
    background: transparent;
}

.viewer-tool svg {
    width: 14px;
    height: 14px;
}

.viewer-tool.is-active {
    border-color: #4c92ff;
    color: #ffffff;
    background: #1f5fc7;
}

.viewer-tool.is-active:hover {
    border-color: var(--atlas-coral);
    color: var(--atlas-coral);
}

.viewer-tool[disabled] {
    opacity: 0.35;
    cursor: not-allowed;
}

#main-image {
    min-height: 620px;
    border: 0 !important;
    border-radius: 0 !important;
    background: #dfe7ed !important;
    box-shadow: none !important;
}

#main-image .image-container {
    overflow: hidden !important;
    position: relative !important;
    cursor: grab;
    touch-action: none;
}

#main-image .image-container.is-panning {
    cursor: grabbing;
}

#main-image img {
    border-radius: 0 !important;
    transform-origin: center center;
    transition: transform 130ms ease-out;
    user-select: none;
}

.viewer-toolbar {
    gap: 4px;
}

.viewer-tool-button {
    display: inline-flex;
    width: 30px;
    height: 30px;
    align-items: center;
    justify-content: center;
    border: 1px solid transparent;
    border-radius: 4px;
    color: #d5e3ef;
    background: transparent;
    cursor: pointer;
}

.viewer-tool-button:hover,
.viewer-tool-button:focus-visible {
    border-color: #4c92ff;
    color: #ffffff;
    background: #1f5fc7;
}

.viewer-tool-button svg {
    width: 16px;
    height: 16px;
}

.viewer-zoom-level {
    min-width: 44px;
    color: #ffffff;
    font: 600 10px/1 var(--atlas-mono);
    text-align: center;
}

#main-image img {
    object-fit: contain !important;
}

.canvas-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 8px 18px;
    min-height: 38px;
    padding: 10px 16px;
    border-top: 1px solid var(--atlas-line);
    color: var(--atlas-muted);
    font: 500 10px/1.5 var(--atlas-mono);
    background: var(--atlas-canvas);
}

.canvas-meta strong {
    color: var(--atlas-ink);
    font-weight: 600;
}

#report-tabs > .tab-nav {
    gap: 20px;
    margin-bottom: 18px;
    border-bottom: 1px solid var(--atlas-line) !important;
    background: transparent !important;
}

.decision-summary {
    padding: 2px 0 19px;
    border-bottom: 1px solid var(--atlas-line);
}

.decision-kicker {
    margin: 0 0 9px;
    color: var(--atlas-muted);
    font: 600 9px/1 var(--atlas-latin);
    letter-spacing: 0.13em;
    text-transform: uppercase;
}

.decision-line {
    display: flex;
    gap: 10px;
    align-items: flex-start;
}

.decision-marker {
    width: 7px;
    height: 7px;
    flex: 0 0 7px;
    margin-top: 7px;
    border-radius: 50%;
    background: var(--atlas-coral);
}

.decision-title {
    margin: 0;
    color: var(--atlas-ink);
    font-size: 20px;
    font-weight: 620;
    letter-spacing: -0.035em;
    line-height: 1.25;
}

.decision-summary-text {
    margin: 10px 0 0 17px;
    color: var(--atlas-muted);
    font-size: 11px;
    line-height: 1.65;
}

.report-meta {
    display: grid;
    grid-template-columns: 1fr 1fr;
    margin: 0;
    padding: 15px 0;
    border-bottom: 1px solid var(--atlas-line);
}

.report-meta div {
    min-width: 0;
    padding: 7px 0;
}

.report-meta dt {
    margin: 0 0 4px;
    color: var(--atlas-muted);
    font-size: 9px;
    letter-spacing: 0.08em;
}

.report-meta dd {
    overflow: hidden;
    margin: 0;
    color: var(--atlas-ink);
    font: 550 11px/1.35 var(--atlas-mono);
    text-overflow: ellipsis;
    white-space: nowrap;
}

.section-label {
    margin: 18px 0 9px;
    color: var(--atlas-ink);
    font-size: 11px;
    font-weight: 650;
}

#key-evidence {
    border: 0 !important;
    border-radius: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}

#key-evidence .grid-wrap {
    padding: 0 !important;
}

#key-evidence .thumbnail-item {
    border: 1px solid var(--atlas-line) !important;
    border-radius: 0 !important;
}

#key-evidence .caption-label {
    color: var(--atlas-muted) !important;
    font: 500 9px/1.25 var(--atlas-mono) !important;
}

.empty-evidence {
    padding: 13px 0;
    border-top: 1px solid var(--atlas-line-soft);
    border-bottom: 1px solid var(--atlas-line-soft);
    color: var(--atlas-muted);
    font-size: 10px;
    line-height: 1.55;
}

.editorial-accordion {
    margin-top: 15px !important;
    border: 0 !important;
    border-top: 1px solid var(--atlas-line) !important;
    border-radius: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}

.editorial-accordion > button {
    min-height: 40px;
    padding-inline: 0 !important;
    color: var(--atlas-ink) !important;
    font-size: 11px !important;
    font-weight: 620 !important;
}

#report-output,
#batch-summary,
#batch-report {
    color: var(--atlas-ink);
    font-size: 11px;
    line-height: 1.7;
}

#report-output h1,
#report-output h2,
#report-output h3,
#batch-report h1,
#batch-report h2,
#batch-report h3 {
    color: var(--atlas-ink) !important;
    font-family: var(--atlas-body) !important;
    font-weight: 620 !important;
    letter-spacing: -0.02em;
}

#report-output h1 {
    font-size: 18px !important;
}

#report-output h2 {
    margin-top: 20px !important;
    font-size: 14px !important;
}

.relation-empty,
.relation-error {
    padding: 14px 0;
    color: var(--atlas-muted);
    font-size: 11px;
    line-height: 1.6;
}

.relation-error {
    color: #9a3f2c;
}

.trace-list {
    margin: 0;
    padding: 0;
    list-style: none;
}

.trace-item {
    position: relative;
    display: grid;
    grid-template-columns: 10px 1fr;
    gap: 9px;
    padding-bottom: 15px;
}

.trace-item:not(:last-child)::before {
    position: absolute;
    top: 9px;
    bottom: -2px;
    left: 3px;
    width: 1px;
    background: var(--atlas-line);
    content: "";
}

.trace-node {
    position: relative;
    z-index: 1;
    width: 7px;
    height: 7px;
    margin-top: 4px;
    border: 1px solid var(--atlas-coral);
    border-radius: 50%;
    background: var(--atlas-canvas);
}

.trace-title {
    display: flex;
    justify-content: space-between;
    gap: 8px;
    color: var(--atlas-ink);
    font-size: 11px;
    font-weight: 600;
}

.trace-time {
    color: var(--atlas-muted);
    font: 500 9px/1.4 var(--atlas-mono);
}

.trace-action {
    margin-top: 3px;
    color: var(--atlas-muted);
    font-size: 10px;
}

.artifact-list {
    margin: 0;
    padding: 0;
    border-top: 1px solid var(--atlas-line);
    list-style: none;
}

.artifact-list li {
    display: flex;
    justify-content: space-between;
    gap: 10px;
    padding: 9px 0;
    border-bottom: 1px solid var(--atlas-line-soft);
    color: var(--atlas-muted);
    font-size: 10px;
}

.artifact-list strong {
    color: var(--atlas-ink);
    font-weight: 600;
}

.secondary-workspace {
    display: grid !important;
    grid-template-columns: minmax(240px, 0.36fr) minmax(0, 0.64fr);
    gap: 0 !important;
    width: min(1160px, calc(100% - 40px));
    margin: 32px auto 54px !important;
    border-top: 1px solid var(--atlas-line);
    border-bottom: 1px solid var(--atlas-line);
}

.secondary-intro,
.secondary-content {
    padding: 28px !important;
}

.secondary-intro {
    border-right: 1px solid var(--atlas-line);
}

.section-heading h3 {
    margin: 0 0 8px !important;
    color: var(--atlas-ink) !important;
    font-size: 19px !important;
    font-weight: 620 !important;
    letter-spacing: -0.025em;
}

.section-heading p {
    margin: 0 0 18px !important;
    color: var(--atlas-muted) !important;
    font-size: 11px !important;
    line-height: 1.65 !important;
}

.prompt-examples ul {
    margin: 10px 0 0 !important;
    padding-left: 17px !important;
    color: var(--atlas-muted) !important;
    font-size: 11px;
}

.prompt-examples li {
    margin: 8px 0 !important;
}

#question-input textarea {
    min-height: 80px !important;
}

.gradio-container textarea,
.gradio-container input {
    border-radius: 1px !important;
}

.result-accordion {
    border-radius: 0 !important;
}

#app-footer {
    display: flex;
    justify-content: space-between;
    gap: 24px;
    padding: 13px 20px;
    border-top: 1px solid var(--atlas-line);
    color: var(--atlas-muted);
    font: 500 9px/1.4 var(--atlas-mono);
    letter-spacing: 0.06em;
}

button:focus-visible,
input:focus-visible,
textarea:focus-visible,
[role="tab"]:focus-visible,
[tabindex]:focus-visible {
    outline: 2px solid var(--atlas-coral) !important;
    outline-offset: 2px !important;
}

footer {
    display: none !important;
}

@media (max-width: 1120px) {
    #atlas-topbar {
        grid-template-columns: minmax(0, 1fr) auto auto;
    }

    #atlas-workspace {
        grid-template-columns: 190px minmax(0, 1fr) 300px;
    }
}

@media (max-width: 900px) {
    #atlas-topbar {
        grid-template-columns: minmax(0, 1fr) auto auto;
    }

    #atlas-workspace {
        display: flex !important;
        min-height: 0;
        flex-direction: column;
    }

    .evidence-rail,
    .report-rail {
        max-height: none;
        border: 0 !important;
        border-bottom: 1px solid var(--atlas-line) !important;
    }

    .evidence-rail {
        display: grid !important;
        grid-template-columns: minmax(150px, 0.55fr) minmax(150px, 0.45fr);
        gap: 14px !important;
    }

    .evidence-rail > :first-child,
    .evidence-rail > :last-child {
        grid-column: 1 / -1;
    }

    .image-stage,
    #main-image {
        min-height: 520px;
    }

    .report-rail {
        padding-inline: max(20px, 7vw) !important;
    }
}

@media (max-width: 620px) {
    #atlas-topbar {
        gap: 8px !important;
        padding-inline: 12px !important;
    }

    .brand-title {
        max-width: 180px;
        font-size: 12px;
    }

    .brand-caption {
        display: none;
    }

    #export-report {
        min-width: 88px !important;
        padding-inline: 10px !important;
    }

    #mode-tabs > .tab-nav {
        overflow-x: auto;
        padding-inline: 12px !important;
    }

    .evidence-rail {
        display: flex !important;
        padding: 16px !important;
        flex-direction: column;
    }

    .image-stage,
    #main-image {
        min-height: 420px;
    }

    .canvas-topline {
        align-items: flex-start;
        padding-block: 10px;
        flex-direction: column;
    }

    .viewer-toolbar {
        width: 100%;
        justify-content: space-between;
        box-shadow: none;
    }

    .secondary-workspace {
        display: flex !important;
        width: calc(100% - 24px);
        flex-direction: column;
    }

    .secondary-intro {
        border-right: 0;
        border-bottom: 1px solid var(--atlas-line);
    }

    .secondary-intro,
    .secondary-content {
        padding: 20px !important;
    }

    #app-footer {
        align-items: flex-start;
        flex-direction: column;
    }
}

@media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
        scroll-behavior: auto !important;
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}
"""

_NODE_LABELS = {
    "intake_agent": "输入检查",
    "vision_perception_agent": "视觉感知",
    "supervisor_agent": "任务路由",
    "scene_analysis_agent": "场景分析",
    "quality_review_agent": "质量复核",
    "ensure_review_agent": "复核确认",
    "decision_scoring_agent": "决策评分",
    "report_agent": "报告生成",
    "persistence_agent": "结果留存",
}

_ACTION_LABELS = {
    "accepted": "输入已接收",
    "analysis_completed": "感知完成",
    "scene_inferred": "场景分析完成",
    "review_completed": "质量复核完成",
    "score_calculated": "评分完成",
    "report_generated": "报告生成",
    "artifacts_saved": "结果已留存",
}

_EMPTY_REPORT = "### 等待分析\n\n上传一张图像并启动分析，结构化报告将在这里呈现。"
_EMPTY_OVERVIEW = """
<section class="decision-summary" aria-live="polite">
    <p class="decision-kicker">Decision note</p>
    <div class="decision-line">
        <span class="decision-marker" aria-hidden="true"></span>
        <h2 class="decision-title">等待分析</h2>
    </div>
    <p class="decision-summary-text">完成图像分析后，这里将展示真实决策与评分摘要。</p>
</section>
<dl class="report-meta">
    <div><dt>分析时间</dt><dd>—</dd></div>
    <div><dt>图像尺寸</dt><dd>—</dd></div>
    <div><dt>检测数量</dt><dd>—</dd></div>
    <div><dt>综合评分</dt><dd>—</dd></div>
</dl>
"""
_EMPTY_RELATION = """
<div class="relation-empty">
    完成分析后，将在这里显示 Agent 执行链、审计时间与已保存产物。
</div>
"""
_UI_ERRORS = (AgentBaseError, OSError, RuntimeError, TypeError, ValueError)


def _empty_workspace_state(image_path: str | None = None) -> Dict[str, Any]:
    """创建会话级工作台快照，避免不同用户或标签页共享结果。"""
    return {
        "image_path": image_path,
        "annotated_path": None,
        "report_overview": _EMPTY_OVERVIEW,
        "report_markdown": _EMPTY_REPORT,
        "relation_html": _EMPTY_RELATION,
        "canvas_meta": _canvas_meta_html(image_path),
        "vision_result": None,
        "run_id": None,
        "report_path": None,
        "chat_history": [],
    }


def _node_label(node: str) -> str:
    return _NODE_LABELS.get(node, node)


def _action_label(action: str) -> str:
    return _ACTION_LABELS.get(action, action)


def _display_value(value: Any, fallback: str = "—") -> str:
    if value is None or value == "":
        return fallback
    return html.escape(str(value))


def _score_text(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "—"
    number = float(value)
    return f"{number:.0f}" if number.is_integer() else f"{number:.1f}"


def _report_overview_html(result: Dict[str, Any]) -> str:
    vision = result.get("vision_result") or {}
    decision = result.get("decision_result") or {}
    decision_code = str(decision.get("decision", "unknown"))
    decision_label = config.DECISION_LABELS.get(decision_code, decision_code)
    summary = decision.get("summary") or "暂无决策摘要。"
    width = vision.get("image_width")
    height = vision.get("image_height")
    dimensions = f"{width} × {height} px" if width and height else "—"
    total_detections = vision.get("total_detections", 0)
    total_score = _score_text(decision.get("total_score"))
    score_level = _display_value(decision.get("score_level"), "未知")
    started_at = _display_value(result.get("started_at"))
    run_id = _display_value(result.get("run_id"))

    return f"""
    <section class="decision-summary" aria-live="polite">
        <p class="decision-kicker">Decision note</p>
        <div class="decision-line">
            <span class="decision-marker" aria-hidden="true"></span>
            <h2 class="decision-title">{html.escape(str(decision_label))}</h2>
        </div>
        <p class="decision-summary-text">{html.escape(str(summary))}</p>
    </section>
    <dl class="report-meta">
        <div><dt>分析时间</dt><dd>{started_at}</dd></div>
        <div><dt>图像尺寸</dt><dd>{dimensions}</dd></div>
        <div><dt>检测数量</dt><dd>{total_detections} 个</dd></div>
        <div><dt>综合评分</dt><dd>{total_score} / 100</dd></div>
        <div><dt>可信等级</dt><dd>{score_level}</dd></div>
        <div><dt>运行编号</dt><dd title="{run_id}">{run_id}</dd></div>
    </dl>
    """


def _canvas_meta_html(
    image_path: str | None,
    result: Dict[str, Any] | None = None,
    *,
    state_label: str = "等待分析",
) -> str:
    file_name = Path(image_path).name if image_path else "尚未选择图像"
    vision = (result or {}).get("vision_result") or {}
    width = vision.get("image_width")
    height = vision.get("image_height")
    dimensions = f"{width} × {height} px" if width and height else "尺寸待读取"
    model = vision.get("model_name") or "模型待运行"
    return (
        '<div class="canvas-meta" aria-live="polite">'
        f"<span><strong>{html.escape(file_name)}</strong></span>"
        f"<span>{html.escape(dimensions)}</span>"
        f"<span>{html.escape(str(model))}</span>"
        f"<span>{html.escape(state_label)}</span>"
        "</div>"
    )


def _relation_html(result: Dict[str, Any]) -> str:
    audit_items = []
    for entry in result.get("audit_log", []):
        node = _node_label(str(entry.get("node", "")))
        action = _action_label(str(entry.get("action", "")))
        entry_time = _display_value(entry.get("time"))
        audit_items.append(
            '<li class="trace-item">'
            '<span class="trace-node" aria-hidden="true"></span>'
            "<div>"
            '<div class="trace-title">'
            f"<span>{html.escape(node)}</span>"
            f'<time class="trace-time">{entry_time}</time>'
            "</div>"
            f'<div class="trace-action">{html.escape(action)}</div>'
            "</div>"
            "</li>"
        )

    files = result.get("output_files") or {}
    artifact_labels = {
        "report": "结构化报告",
        "decision": "决策数据",
        "audit": "审计记录",
        "state": "状态快照",
        "annotated": "检测图像",
    }
    artifact_items = [
        f"<li><span>{label}</span><strong>已保存</strong></li>"
        for key, label in artifact_labels.items()
        if files.get(key)
    ]
    trace_markup = "".join(audit_items) or (
        '<li class="relation-empty">本次运行没有可显示的审计节点。</li>'
    )
    artifact_markup = "".join(artifact_items) or (
        "<li><span>运行产物</span><strong>未生成</strong></li>"
    )
    return (
        '<p class="section-label">执行链路</p>'
        f'<ol class="trace-list">{trace_markup}</ol>'
        '<p class="section-label">结果产物</p>'
        f'<ul class="artifact-list">{artifact_markup}</ul>'
    )


def _key_evidence_items(result: Dict[str, Any]) -> list[tuple[Image.Image, str]]:
    vision = result.get("vision_result") or {}
    image_path = vision.get("image_path")
    if not image_path:
        return []

    candidates = [
        item
        for item in vision.get("detections", [])
        if isinstance(item, dict) and isinstance(item.get("bbox"), dict)
    ]
    candidates.sort(key=lambda item: float(item.get("confidence", 0)), reverse=True)
    if not candidates:
        return []

    try:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
    except (OSError, UnidentifiedImageError) as error:
        _log.warning("关键证据缩略图读取失败: %s", error)
        return []

    evidence: list[tuple[Image.Image, str]] = []
    for item in candidates:
        bbox = item["bbox"]
        try:
            x1 = max(0, min(image.width, int(float(bbox["x1"]))))
            y1 = max(0, min(image.height, int(float(bbox["y1"]))))
            x2 = max(0, min(image.width, int(float(bbox["x2"]))))
            y2 = max(0, min(image.height, int(float(bbox["y2"]))))
        except (KeyError, TypeError, ValueError):
            continue
        if x2 <= x1 or y2 <= y1:
            continue

        padding = max(4, int(max(x2 - x1, y2 - y1) * 0.12))
        crop = image.crop(
            (
                max(0, x1 - padding),
                max(0, y1 - padding),
                min(image.width, x2 + padding),
                min(image.height, y2 + padding),
            )
        )
        class_name = str(item.get("class_name", "目标"))
        confidence = float(item.get("confidence", 0))
        evidence.append((crop, f"{class_name} · {confidence:.0%}"))
        if len(evidence) == 3:
            break
    return evidence


def _download_update(report_path: str | None) -> gr.DownloadButton:
    return gr.DownloadButton(
        label="导出报告",
        value=report_path,
        variant="primary",
        interactive=bool(report_path),
        elem_id="export-report",
    )


def reset_workspace(image_path: str | None):
    """Reset derived outputs when the user selects a different image."""
    return (
        image_path,
        None,
        _EMPTY_OVERVIEW,
        _EMPTY_REPORT,
        [],
        _EMPTY_RELATION,
        _canvas_meta_html(image_path),
        "",
        _download_update(None),
        _empty_workspace_state(image_path),
    )


def run_with_ui(image_path: str):
    """Run the local workflow and map its real outputs to the workbench."""
    if not image_path:
        return reset_workspace(None)

    _log.info("Gradio 请求: %s", image_path)

    try:
        result = run_pipeline(image_path)

        report = result.get("final_report", "")
        files = result.get("output_files", {})
        annotated_path = files.get("annotated") or image_path
        report_path = files.get("report")
        run_id = str(result.get("run_id", "unknown"))
        get_or_create_chat(run_id, result)
        vision_result = result.get("vision_result") or {}
        overview_html = _report_overview_html(result)
        relation_html = _relation_html(result)
        canvas_meta = _canvas_meta_html(image_path, result, state_label="分析完成")

        chat_status = (
            f"对话已就绪。切换到“智能追问”可基于本次结果继续提问。\n"
            f"运行编号: {run_id}"
        )

        _log.info("Gradio 分析完成: %s", run_id)
        return (
            annotated_path,
            annotated_path,
            overview_html,
            report,
            _key_evidence_items(result),
            relation_html,
            canvas_meta,
            chat_status,
            _download_update(report_path),
            {
                "image_path": image_path,
                "annotated_path": annotated_path,
                "report_overview": overview_html,
                "report_markdown": report,
                "relation_html": relation_html,
                "canvas_meta": canvas_meta,
                "vision_result": vision_result,
                "run_id": run_id,
                "report_path": report_path,
                "chat_history": [],
            },
        )

    except _UI_ERRORS as error:
        _log.exception("Gradio 分析失败")
        return (
            image_path,
            None,
            (
                '<section class="decision-summary" aria-live="polite">'
                '<p class="decision-kicker">Decision note</p>'
                '<div class="decision-line">'
                '<span class="decision-marker" aria-hidden="true"></span>'
                '<h2 class="decision-title">分析未完成</h2>'
                "</div>"
                f'<p class="decision-summary-text">{html.escape(str(error))}</p>'
                "</section>"
            ),
            f"### 分析未完成\n\n{type(error).__name__}：{error}\n\n"
            "请检查图片格式、模型权重或运行配置后重试。",
            [],
            (
                '<div class="relation-error">'
                f"运行失败：{html.escape(str(error))}"
                "</div>"
            ),
            _canvas_meta_html(image_path, state_label="分析失败"),
            "",
            _download_update(None),
            _empty_workspace_state(image_path),
        )


def restore_workspace(workspace_state: Dict[str, Any] | None):
    """标签切回工作台时，从当前会话快照完整恢复分析结果。"""
    state = workspace_state or _empty_workspace_state()
    image_path = state.get("image_path")
    annotated_path = state.get("annotated_path")
    if not image_path or not annotated_path:
        return reset_workspace(image_path)[:-1]

    vision_result = state.get("vision_result") or {}
    run_id = state.get("run_id", "unknown")
    return (
        annotated_path,
        annotated_path,
        state.get("report_overview", _EMPTY_OVERVIEW),
        state.get("report_markdown", _EMPTY_REPORT),
        _key_evidence_items({"vision_result": vision_result}),
        state.get("relation_html", _EMPTY_RELATION),
        state.get("canvas_meta", _canvas_meta_html(image_path)),
        f"对话已就绪。运行编号: {run_id}",
        _download_update(state.get("report_path")),
    )


def restore_chat(workspace_state: Dict[str, Any] | None):
    """标签切回智能追问时恢复同一会话的对话记录。"""
    return (workspace_state or {}).get("chat_history", [])


def chat_followup(
    question: str,
    chat_history: list,
    workspace_state: Dict[str, Any] | None,
):
    """处理对话追问，并将模型输出实时推送到 Gradio 对话框。"""
    chat_history = list(chat_history or [])
    state = dict(workspace_state or _empty_workspace_state())

    if not question or not question.strip():
        yield chat_history, "", state
        return

    run_id = state.get("run_id", "")
    chat = get_or_create_chat(run_id)

    if chat is None:
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": "请先在“单图分析”中完成一次分析。"})
        state["chat_history"] = chat_history
        yield chat_history, "", state
        return

    chat_history.append({"role": "user", "content": question})
    chat_history.append({"role": "assistant", "content": ""})
    state["chat_history"] = chat_history
    yield chat_history, "", state

    for partial_answer in chat.ask_stream(question):
        chat_history[-1] = {"role": "assistant", "content": partial_answer}
        state["chat_history"] = chat_history
        yield chat_history, "", state


def batch_analyze(files: list, progress=gr.Progress()):
    """批量分析多张图片。"""
    if not files:
        return "请上传至少一张图片", ""

    results_summary = []
    all_reports = []

    total = len(files)
    for i, file_path in enumerate(files):
        file_path = str(file_path)
        progress(i / total, desc=f"正在分析 ({i+1}/{total}): {os.path.basename(file_path)}")
        try:
            result = run_pipeline(file_path)
            run_id = result.get("run_id", "unknown")
            vision = result.get("vision_result", {})
            scene = result.get("scene_result", {})

            results_summary.append(
                f"| {os.path.basename(file_path)} | {run_id} | "
                f"{vision.get('total_detections', 0)} 个目标 | "
                f"{scene.get('scene_type', '未分类')} |"
            )
            all_reports.append(
                f"---\n### {os.path.basename(file_path)} (run_id: {run_id})\n\n"
                + result.get("final_report", "")
            )
        except Exception as e:
            results_summary.append(
                f"| {os.path.basename(file_path)} | 失败 | {e} | 未分类 |"
            )
            all_reports.append(f"---\n### {os.path.basename(file_path)}\n\n分析失败: {e}")

    progress(1.0, desc="分析完成")

    summary_table = (
        "| 文件名 | run_id | 检测结果 | 场景类型 |\n"
        "|--------|--------|----------|----------|\n"
        + "\n".join(results_summary)
    )

    full_report = "## 批量分析报告\n\n" + summary_table + "\n\n" + "\n".join(all_reports)

    return summary_table, full_report


# ===== Gradio 界面 =====

_app_name_html = html.escape(APP_NAME)
_TOOLBAR_HTML = """
<div class="viewer-toolbar" role="toolbar" aria-label="图像查看工具">
    <span class="viewer-tool is-active" role="status" title="按住左键拖拽移动图像" aria-label="拖拽移动图像">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" aria-hidden="true">
            <path d="M8 11V6.8a1.7 1.7 0 0 1 3.4 0V10 5.8a1.7 1.7 0 0 1 3.4 0V10 7.2a1.7 1.7 0 0 1 3.4 0v5.5c0 5-2.8 7.3-7 7.3-2.7 0-4.2-1.6-5.5-3.8L3.9 13a1.7 1.7 0 0 1 2.9-1.8L8 13.1V11Z"/>
        </svg>
    </span>
    <div class="viewer-tool-button" role="button" tabindex="0" data-canvas-action="zoom-out" title="缩小图像" aria-label="缩小图像">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" aria-hidden="true">
            <circle cx="11" cy="11" r="6.5"/><path d="M8 11h6m2 4 4 4"/>
        </svg>
    </div>
    <div class="viewer-tool-button" role="button" tabindex="0" data-canvas-action="reset" title="适应画布" aria-label="适应画布">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" aria-hidden="true">
            <path d="M4 9V4h5M20 9V4h-5M4 15v5h5m11-5v5h-5"/>
        </svg>
    </div>
    <span class="viewer-zoom-level" id="viewer-zoom-level" aria-live="polite">100%</span>
    <div class="viewer-tool-button" role="button" tabindex="0" data-canvas-action="zoom-in" title="放大图像" aria-label="放大图像">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" aria-hidden="true">
            <circle cx="11" cy="11" r="6.5"/><path d="M8 11h6m-3-3v6m5 1 4 4"/>
        </svg>
    </div>
</div>
"""

_VIEWER_SCRIPT = """
<script>
(() => {
  if (window.__visualDecisionCanvasReady) return;
  window.__visualDecisionCanvasReady = true;
  const state = window.__visualDecisionCanvas = { scale: 1, x: 0, y: 0, source: "", dragging: false };
  const minScale = 0.25, maxScale = 4, step = 0.25;
  const viewer = () => document.querySelector('#main-image .image-container');
  const image = () => document.querySelector('#main-image img');
  const clamp = (value, limit) => Math.max(-limit, Math.min(limit, value));
  const render = () => {
    const stage = viewer(), img = image(), label = document.querySelector('#viewer-zoom-level');
    if (!stage || !img) return;
    const limitX = Math.max(0, (stage.clientWidth * state.scale - stage.clientWidth) / 2);
    const limitY = Math.max(0, (stage.clientHeight * state.scale - stage.clientHeight) / 2);
    state.x = clamp(state.x, limitX); state.y = clamp(state.y, limitY);
    img.style.transform = `translate3d(${state.x}px, ${state.y}px, 0) scale(${state.scale})`;
    if (label) label.textContent = `${Math.round(state.scale * 100)}%`;
  };
  const reset = (force = false) => {
    const img = image();
    if (!img) return;
    const source = img.currentSrc || img.src || "";
    if (force || state.source !== source) {
      state.scale = 1; state.x = 0; state.y = 0; state.source = source;
    }
    render();
  };
  document.addEventListener('click', (event) => {
    const button = event.target.closest('[data-canvas-action]');
    if (!button) return;
    if (button.dataset.canvasAction === 'zoom-in') state.scale = Math.min(maxScale, state.scale + step);
    if (button.dataset.canvasAction === 'zoom-out') state.scale = Math.max(minScale, state.scale - step);
    if (button.dataset.canvasAction === 'reset') { state.scale = 1; state.x = 0; state.y = 0; }
    render();
  });
  document.addEventListener('keydown', (event) => {
    const button = event.target.closest?.('[data-canvas-action]');
    if (button && (event.key === 'Enter' || event.key === ' ')) {
      event.preventDefault(); button.click(); return;
    }
    if (event.target.matches('input, textarea, [contenteditable="true"]')) return;
    if (event.key === '+' || event.key === '=') state.scale = Math.min(maxScale, state.scale + step);
    else if (event.key === '-') state.scale = Math.max(minScale, state.scale - step);
    else if (event.key === '0') { state.scale = 1; state.x = 0; state.y = 0; }
    else return;
    event.preventDefault(); render();
  });
  document.addEventListener('wheel', (event) => {
    const stage = viewer();
    if (!stage || !stage.contains(event.target)) return;
    event.preventDefault();
    state.scale = Math.max(minScale, Math.min(maxScale, state.scale * (event.deltaY < 0 ? 1.1 : 0.9)));
    render();
  }, { passive: false });
  document.addEventListener('pointerdown', (event) => {
    const stage = viewer();
    if (!stage || !stage.contains(event.target) || event.button !== 0) return;
    state.dragging = true; state.startX = event.clientX - state.x; state.startY = event.clientY - state.y;
    stage.classList.add('is-panning'); stage.setPointerCapture?.(event.pointerId);
  });
  document.addEventListener('pointermove', (event) => {
    if (!state.dragging) return;
    state.x = event.clientX - state.startX; state.y = event.clientY - state.startY; render();
  });
  document.addEventListener('pointerup', () => { state.dragging = false; viewer()?.classList.remove('is-panning'); });
  new MutationObserver(() => requestAnimationFrame(() => reset(false))).observe(document.body, { childList: true, subtree: true });
  window.addEventListener('resize', render);
  requestAnimationFrame(() => reset(false));
})();
</script>
"""

with gr.Blocks(
    title=APP_NAME,
    fill_width=True,
) as demo:
    workspace_state = gr.State(_empty_workspace_state())
    with gr.Row(elem_id="atlas-topbar"):
        gr.HTML(
            f"""
            <div class="atlas-brand" aria-label="{_app_name_html}">
                <span class="atlas-mark" aria-hidden="true"></span>
                <div class="brand-copy">
                    <p class="brand-title">{_app_name_html}</p>
                    <p class="brand-caption">Visual analysis workspace</p>
                </div>
            </div>
            """
        )
        export_btn = gr.DownloadButton(
            "导出报告",
            value=None,
            variant="primary",
            interactive=False,
            size="sm",
            elem_id="export-report",
        )
        gr.HTML(
            """
            <button class="more-action" type="button" aria-label="暂无更多操作"
                    title="暂无更多操作" disabled>
                <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                    <circle cx="5" cy="12" r="1.7"/>
                    <circle cx="12" cy="12" r="1.7"/>
                    <circle cx="19" cy="12" r="1.7"/>
                </svg>
            </button>
            """
        )

    with gr.Tabs(elem_id="mode-tabs"):
        with gr.TabItem("分析工作台") as workbench_tab:
            with gr.Row(elem_id="atlas-workspace"):
                with gr.Column(elem_classes=["evidence-rail"], min_width=220):
                    gr.HTML(
                        """
                        <div>
                            <h2 class="rail-heading">证据序列</h2>
                            <p class="rail-intro">当前图像与检测结果构成本次分析证据。</p>
                        </div>
                        """
                    )
                    gr.HTML(
                        """
                        <div class="evidence-entry-head">
                            <span class="evidence-name">当前图像</span>
                            <span class="evidence-state">Source</span>
                        </div>
                        """
                    )
                    image_input = gr.Image(
                        type="filepath",
                        sources=["upload"],
                        show_label=False,
                        height=148,
                        buttons=[],
                        placeholder="拖放或点击上传图像",
                        elem_id="image-input",
                        elem_classes=["evidence-image", "is-current"],
                    )
                    gr.HTML(
                        """
                        <div class="evidence-entry-head evidence-entry">
                            <span class="evidence-name">检测结果</span>
                            <span class="evidence-state">Result</span>
                        </div>
                        """
                    )
                    evidence_result_output = gr.Image(
                        type="filepath",
                        show_label=False,
                        height=132,
                        interactive=False,
                        buttons=[],
                        placeholder="分析后生成",
                        elem_classes=["evidence-image"],
                    )
                    run_btn = gr.Button(
                        "开始分析",
                        variant="secondary",
                        elem_id="run-button",
                    )
                    gr.Markdown(
                        "支持常见图片格式。结果与审计记录保存在本地任务目录。",
                        elem_classes=["support-note"],
                    )

                with gr.Column(elem_classes=["image-stage"], min_width=420):
                    gr.HTML(
                        f"""
                        <div class="canvas-topline">
                            <div class="canvas-heading-wrap">
                                <span class="canvas-status-dot" aria-hidden="true"></span>
                                <h2 class="canvas-heading">图像审阅</h2>
                            </div>
                            {_TOOLBAR_HTML}
                        </div>
                        """
                    )
                    main_image_output = gr.Image(
                        type="filepath",
                        show_label=False,
                        height=680,
                        interactive=False,
                        buttons=["fullscreen", "download"],
                        placeholder="从左侧上传图像以开始审阅",
                        elem_id="main-image",
                    )
                    canvas_meta_output = gr.HTML(_canvas_meta_html(None))

                with gr.Column(elem_classes=["report-rail"], min_width=320):
                    with gr.Tabs(elem_id="report-tabs"):
                        with gr.TabItem("报告"):
                            report_overview = gr.HTML(_EMPTY_OVERVIEW)
                            gr.HTML('<p class="section-label">关键证据</p>')
                            key_evidence_output = gr.Gallery(
                                value=[],
                                show_label=False,
                                columns=3,
                                rows=1,
                                height=116,
                                allow_preview=True,
                                object_fit="cover",
                                buttons=[],
                                type="pil",
                                elem_id="key-evidence",
                            )
                            gr.HTML(
                                """
                                <div class="empty-evidence">
                                    最多展示三个带检测框的高置信度目标；没有可靠目标时保持为空。
                                </div>
                                """
                            )
                            with gr.Accordion(
                                "完整结构化报告",
                                open=False,
                                elem_classes=["editorial-accordion"],
                            ):
                                report_output = gr.Markdown(
                                    _EMPTY_REPORT,
                                    elem_id="report-output",
                                )
                        with gr.TabItem("关联"):
                            relation_output = gr.HTML(_EMPTY_RELATION)

            chat_status_indicator = gr.Textbox(
                label="对话状态",
                interactive=False,
                visible=False,
            )

            workspace_outputs = [
                main_image_output,
                evidence_result_output,
                report_overview,
                report_output,
                key_evidence_output,
                relation_output,
                canvas_meta_output,
                chat_status_indicator,
                export_btn,
            ]
            workspace_outputs_with_state = [*workspace_outputs, workspace_state]
            image_input.change(
                fn=reset_workspace,
                inputs=image_input,
                outputs=workspace_outputs_with_state,
            )
            run_btn.click(
                fn=run_with_ui,
                inputs=image_input,
                outputs=workspace_outputs_with_state,
            )

        with gr.TabItem("智能追问") as chat_tab:
            with gr.Row(elem_classes=["secondary-workspace"]):
                with gr.Column(elem_classes=["secondary-intro"]):
                    gr.Markdown(
                        """
                        ### 继续追问
                        完成一次单图分析后，可针对检测结果、场景判断和评分依据继续提问。
                        """,
                        elem_classes=["section-heading"],
                    )
                    gr.Markdown(
                        """
                        **可以这样问**

                        - 为什么建议人工复核？
                        - 哪些类别的置信度较低？
                        - 评分由哪些证据组成？
                        - 降低检测阈值可能带来什么影响？
                        """,
                        elem_classes=["prompt-examples"],
                    )
                with gr.Column(elem_classes=["secondary-content"]):
                    chatbot = gr.Chatbot(
                        label="对话记录",
                        height=430,
                        buttons=["copy"],
                    )
                    question_input = gr.Textbox(
                        label="你的问题",
                        placeholder="针对本次分析结果继续提问",
                        lines=2,
                        elem_id="question-input",
                    )
                    with gr.Row():
                        send_btn = gr.Button(
                            "发送问题",
                            variant="secondary",
                            elem_id="send-button",
                        )
                        clear_btn = gr.Button(
                            "清空对话",
                            variant="secondary",
                            elem_classes=["secondary-action"],
                        )

                    send_btn.click(
                        fn=chat_followup,
                        inputs=[question_input, chatbot, workspace_state],
                        outputs=[chatbot, question_input, workspace_state],
                    )
                    question_input.submit(
                        fn=chat_followup,
                        inputs=[question_input, chatbot, workspace_state],
                        outputs=[chatbot, question_input, workspace_state],
                    )

                    def clear_chat_history(workspace_state: Dict[str, Any] | None):
                        state = dict(workspace_state or _empty_workspace_state())
                        run_id = state.get("run_id", "")
                        if run_id:
                            clear_chat(run_id)
                        state["chat_history"] = []
                        return [], state

                    clear_btn.click(
                        fn=clear_chat_history,
                        inputs=workspace_state,
                        outputs=[chatbot, workspace_state],
                    )

        with gr.TabItem("批量处理"):
            with gr.Row(elem_classes=["secondary-workspace"]):
                with gr.Column(elem_classes=["secondary-intro"]):
                    gr.Markdown(
                        """
                        ### 批量图像分析
                        一次上传多张图片，系统将逐张处理并汇总结果。
                        """,
                        elem_classes=["section-heading"],
                    )
                    batch_files = gr.File(
                        label="选择多张图片",
                        file_count="multiple",
                        file_types=["image"],
                    )
                    batch_btn = gr.Button(
                        "开始批量分析",
                        variant="secondary",
                        elem_id="batch-run-button",
                    )
                with gr.Column(elem_classes=["secondary-content"]):
                    gr.Markdown(
                        """
                        ### 汇总与报告
                        先查看总体统计，再展开每张图片的完整报告。
                        """,
                        elem_classes=["section-heading"],
                    )
                    batch_summary = gr.Markdown(
                        "### 等待文件\n\n选择图片并启动批量分析。",
                        elem_id="batch-summary",
                    )
                    with gr.Accordion(
                        "完整批量报告",
                        open=False,
                        elem_classes=["result-accordion"],
                    ):
                        batch_report = gr.Markdown(
                            "完成分析后显示详细报告。",
                            elem_id="batch-report",
                        )

            batch_btn.click(
                fn=batch_analyze,
                inputs=batch_files,
                outputs=[batch_summary, batch_report],
            )

    workbench_tab.select(
        fn=restore_workspace,
        inputs=workspace_state,
        outputs=workspace_outputs,
    )
    chat_tab.select(
        fn=restore_chat,
        inputs=workspace_state,
        outputs=chatbot,
    )

    gr.HTML(
        f"""
        <footer id="app-footer">
            <span>{_app_name_html}</span>
            <span>YOLOv8 · LANGGRAPH · TRACEABLE DECISION</span>
        </footer>
        """
    )


def launch_app() -> None:
    port_text = os.getenv("APP_PORT") or os.getenv("GRADIO_SERVER_PORT", "7861")
    try:
        server_port = int(port_text)
    except ValueError as error:
        raise ValueError(f"APP_PORT 必须是整数，当前值为 {port_text!r}") from error

    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=server_port,
        share=os.getenv("GRADIO_SHARE", "false").lower() == "true",
        theme=APP_THEME,
        css=APP_CSS,
        head=_VIEWER_SCRIPT,
    )


if __name__ == "__main__":
    launch_app()
