from fastapi.testclient import TestClient
from langchain_core.documents import Document

import prodml.api.rag_api as rag_api

client = TestClient(rag_api.app)


def fake_ask(question):
    return {
        "answer": "إجابة تجريبية",
        "context": [
            Document(
                page_content="نص",
                metadata={"citation": "Egyptian Civil Code, Article 147"},
            )
        ],
    }


class FakeDB:
    class _collection:
        @staticmethod
        def count():
            return 1133


def test_health(monkeypatch):
    monkeypatch.setattr(rag_api, "get_db", lambda: FakeDB())
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["documents_indexed"] == 1133


def test_ask_ok(monkeypatch):
    monkeypatch.setattr(rag_api, "ask_legal_question", fake_ask)
    r = client.post("/ask", json={"question": "ما هو البيع؟"})
    assert r.status_code == 200
    assert r.json()["sources"] == ["Egyptian Civil Code, Article 147"]


def test_ask_empty_is_422():
    r = client.post("/ask", json={"question": "   "})
    assert r.status_code == 422
