"""API tests — ticket path only (no torch required)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.ticket_classification import train_ticket_classifier


@pytest.fixture(scope="module")
def client():
    train_ticket_classifier(save=True)
    from src.api import app

    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["tickets_ready"] is True


def test_classify_ticket(client):
    r = client.post(
        "/tickets/classify",
        json={"text": "Payment was charged twice, need a refund"},
    )
    assert r.status_code == 200
    body = r.json()
    assert "category" in body
    assert "priority" in body
    assert "confidence" in body
    assert body["method"] in {"tfidf-logreg", "rule-based"}


def test_classify_batch(client):
    r = client.post(
        "/tickets/classify/batch",
        json={
            "texts": [
                "Cannot login to my account",
                "App keeps crashing on startup",
            ]
        },
    )
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 2


def test_empty_text_rejected(client):
    r = client.post("/tickets/classify", json={"text": ""})
    assert r.status_code == 422
