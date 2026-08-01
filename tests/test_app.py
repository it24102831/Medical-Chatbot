import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture()
def app_module(monkeypatch):
    monkeypatch.setenv("PINECONE_API_KEY", "pc_test_key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or_test_key")
    monkeypatch.setenv("PINECONE_INDEX_NAME", "medical-chatbot-v2")
    monkeypatch.setenv("PINECONE_NAMESPACE", "medical-book")
    monkeypatch.setenv("MAX_INPUT_CHARS", "2000")
    mod = importlib.import_module("app")
    mod = importlib.reload(mod)
    mod._pinecone_client = None
    mod._pinecone_index = None
    mod._http_session = None
    return mod


@pytest.fixture()
def client(app_module):
    return app_module.app.test_client()


def test_import_app_with_placeholder_config(monkeypatch):
    monkeypatch.setenv("PINECONE_API_KEY", "YOUR_PINECONE_API_KEY")
    monkeypatch.setenv("OPENROUTER_API_KEY", "YOUR_OPENROUTER_API_KEY")
    mod = importlib.import_module("app")
    importlib.reload(mod)


def test_get_index(client):
    res = client.get("/")
    assert res.status_code == 200


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.get_json()["status"] == "healthy"


def test_ready_success(client, app_module, monkeypatch):
    class FakePc:
        def describe_index(self, name):
            return {"status": {"ready": True}}

    monkeypatch.setattr(app_module, "get_pinecone_client", lambda: FakePc())
    res = client.get("/ready")
    assert res.status_code == 200
    assert res.get_json()["status"] == "ready"


def test_ready_failure(client, app_module, monkeypatch):
    monkeypatch.setattr(app_module, "get_pinecone_client", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    res = client.get("/ready")
    assert res.status_code == 503
    assert "error" in res.get_json()


def test_post_get_form_success(client, app_module, monkeypatch):
    monkeypatch.setattr(app_module, "retrieve_context", lambda q: "ctx")
    monkeypatch.setattr(app_module, "generate_answer", lambda q, c: "ok")
    res = client.post("/get", data={"msg": "hello"})
    assert res.status_code == 200
    assert res.get_json() == {"answer": "ok"}


def test_post_get_json_success(client, app_module, monkeypatch):
    monkeypatch.setattr(app_module, "retrieve_context", lambda q: "ctx")
    monkeypatch.setattr(app_module, "generate_answer", lambda q, c: "ok json")
    res = client.post("/get", json={"msg": "hello"})
    assert res.status_code == 200
    assert res.get_json() == {"answer": "ok json"}


def test_post_get_empty_question(client):
    res = client.post("/get", json={"msg": "   "})
    assert res.status_code == 400
    assert "error" in res.get_json()


def test_post_get_oversized_question(client):
    res = client.post("/get", json={"msg": "a" * 2001})
    assert res.status_code == 413
    assert "error" in res.get_json()


def test_successful_retrieval_and_answer_generation(client, app_module, monkeypatch):
    def fake_retrieve(q):
        assert q == "question"
        return "retrieved context"

    def fake_answer(q, c):
        assert c == "retrieved context"
        return "final answer"

    monkeypatch.setattr(app_module, "retrieve_context", fake_retrieve)
    monkeypatch.setattr(app_module, "generate_answer", fake_answer)
    res = client.post("/get", json={"msg": "question"})
    assert res.status_code == 200
    assert res.get_json()["answer"] == "final answer"


def test_no_pinecone_matches(client, app_module, monkeypatch):
    seen = {}

    def fake_answer(q, c):
        seen["context"] = c
        return "fallback"

    monkeypatch.setattr(app_module, "retrieve_context", lambda q: "")
    monkeypatch.setattr(app_module, "generate_answer", fake_answer)
    res = client.post("/get", json={"msg": "question"})
    assert res.status_code == 200
    assert seen["context"] == ""


def test_pinecone_authentication_failure(client, app_module, monkeypatch):
    monkeypatch.setattr(app_module, "retrieve_context", lambda q: (_ for _ in ()).throw(PermissionError("Pinecone authentication failed.")))
    res = client.post("/get", json={"msg": "hello"})
    assert res.status_code == 502
    assert "error" in res.get_json()


def test_pinecone_timeout(client, app_module, monkeypatch):
    monkeypatch.setattr(app_module, "retrieve_context", lambda q: (_ for _ in ()).throw(TimeoutError("Pinecone request timed out.")))
    res = client.post("/get", json={"msg": "hello"})
    assert res.status_code == 504


def test_openrouter_authentication_failure(client, app_module, monkeypatch):
    monkeypatch.setattr(app_module, "retrieve_context", lambda q: "ctx")
    monkeypatch.setattr(app_module, "generate_answer", lambda q, c: (_ for _ in ()).throw(PermissionError("OpenRouter authentication failed.")))
    res = client.post("/get", json={"msg": "hello"})
    assert res.status_code == 502


def test_openrouter_rate_limit(client, app_module, monkeypatch):
    monkeypatch.setattr(app_module, "retrieve_context", lambda q: "ctx")
    monkeypatch.setattr(app_module, "generate_answer", lambda q, c: (_ for _ in ()).throw(RuntimeError("OpenRouter rate limit reached.")))
    res = client.post("/get", json={"msg": "hello"})
    assert res.status_code == 502


def test_openrouter_timeout(client, app_module, monkeypatch):
    monkeypatch.setattr(app_module, "retrieve_context", lambda q: "ctx")
    monkeypatch.setattr(app_module, "generate_answer", lambda q, c: (_ for _ in ()).throw(TimeoutError("OpenRouter request timed out.")))
    res = client.post("/get", json={"msg": "hello"})
    assert res.status_code == 504


def test_malformed_external_response(app_module, monkeypatch):
    class FakeResponse:
        status_code = 200

        def json(self):
            return {"choices": []}

    monkeypatch.setattr(app_module, "_openrouter_chat_completion", lambda messages: FakeResponse())
    with pytest.raises(ValueError):
        app_module.generate_answer("q", "ctx")


def test_response_schema_error(client, app_module, monkeypatch):
    monkeypatch.setattr(app_module, "retrieve_context", lambda q: (_ for _ in ()).throw(RuntimeError("failed")))
    res = client.post("/get", json={"msg": "hello"})
    data = res.get_json()
    assert set(data.keys()) == {"error"}


def test_response_schema_success(client, app_module, monkeypatch):
    monkeypatch.setattr(app_module, "retrieve_context", lambda q: "ctx")
    monkeypatch.setattr(app_module, "generate_answer", lambda q, c: "ok")
    res = client.post("/get", json={"msg": "hello"})
    data = res.get_json()
    assert set(data.keys()) == {"answer"}


def test_repeated_requests_reuse_clients(app_module, monkeypatch):
    call_count = {"pinecone": 0}

    class FakePinecone:
        def __init__(self, api_key):
            call_count["pinecone"] += 1

    monkeypatch.setattr(app_module, "Pinecone", FakePinecone)
    app_module._pinecone_client = None
    one = app_module.get_pinecone_client()
    two = app_module.get_pinecone_client()
    assert one is two
    assert call_count["pinecone"] == 1


def test_unsupported_content_type(client):
    res = client.post("/get", data="msg=hello", headers={"Content-Type": "text/plain"})
    assert res.status_code == 415
