import chat_agent


class _Chunk:
    def __init__(self, content):
        self.choices = [type("Choice", (), {"delta": type("Delta", (), {"content": content})()})()]


class _FakeCompletions:
    def create(self, **kwargs):
        assert kwargs["stream"] is True
        return [_Chunk("流式"), _Chunk(None), _Chunk("回答")]


class _FakeClient:
    def __init__(self, **kwargs):
        self.chat = type("Chat", (), {"completions": _FakeCompletions()})()


def test_ask_stream_yields_incremental_content_and_persists_history(monkeypatch):
    monkeypatch.setattr(chat_agent, "OpenAI", _FakeClient)
    monkeypatch.setattr(chat_agent, "_OPENAI_AVAILABLE", True)
    chat = chat_agent.AnalysisChat({"run_id": "stream-test"})

    updates = list(chat.ask_stream("测试问题"))

    assert updates == ["流式", "流式回答"]
    assert chat.get_history() == [
        {"role": "user", "content": "测试问题"},
        {"role": "assistant", "content": "流式回答"},
    ]
