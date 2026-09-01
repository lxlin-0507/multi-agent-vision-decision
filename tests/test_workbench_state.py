import app


def test_restore_workspace_rehydrates_saved_analysis_without_rerun():
    state = {
        "image_path": "/tmp/source.jpg",
        "annotated_path": "/tmp/annotated.jpg",
        "report_overview": "<section>结果可用</section>",
        "report_markdown": "# 已完成报告",
        "relation_html": "<div>执行链路</div>",
        "canvas_meta": "<div>分析完成</div>",
        "vision_result": {"total_detections": 2},
        "run_id": "run-state-test",
        "report_path": "/tmp/report.md",
        "chat_history": [{"role": "user", "content": "测试"}],
    }

    restored = app.restore_workspace(state)

    assert len(restored) == 9
    assert restored[0] == "/tmp/annotated.jpg"
    assert restored[1] == "/tmp/annotated.jpg"
    assert restored[3] == "# 已完成报告"
    assert "run-state-test" in restored[7]
    assert app.restore_chat(state) == state["chat_history"]
