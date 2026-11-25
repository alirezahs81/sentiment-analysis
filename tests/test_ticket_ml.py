"""Tests for the ML ticket classifier and dataset loading."""

import pandas as pd

from src.ticket_classification import (
    TicketClassifier,
    load_ticket_data,
    TICKET_CATEGORIES,
)


def test_bundled_ticket_dataset():
    df = load_ticket_data()
    assert isinstance(df, pd.DataFrame)
    assert {"ticket_text", "category"} <= set(df.columns)
    assert len(df) >= 50
    assert set(df["category"]) <= set(TICKET_CATEGORIES.values())


def test_train_ml_and_classify():
    df = load_ticket_data()
    classifier = TicketClassifier()
    scores = classifier.train_ml(df, text_col="ticket_text", label_col="category")
    assert scores is not None
    assert classifier.model_name == "tfidf-logreg"

    result = classifier.classify("I was charged twice and need a refund now")
    assert result["method"] == "tfidf-logreg"
    assert result["category"] == "Billing & Payments"
    assert 0.0 <= result["confidence"] <= 1.0


def test_fallback_to_rules_without_ml():
    classifier = TicketClassifier()  # no ML training
    result = classifier.classify("Forgot password, can't access account")
    assert result["method"] == "rule-based"


def test_train_ml_with_tiny_data_keeps_rules():
    classifier = TicketClassifier()
    tiny = pd.DataFrame({
        "ticket_text": ["a", "b"],
        "category": ["Technical Issues", "Technical Issues"],
    })
    scores = classifier.train_ml(tiny, text_col="ticket_text", label_col="category")
    assert scores is None
    assert classifier.model_name == "rule-based"


def test_save_and_load_ml():
    import tempfile
    from pathlib import Path

    df = load_ticket_data()
    classifier = TicketClassifier()
    classifier.train_ml(df, text_col="ticket_text", label_col="category")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ticket_classifier.joblib"
        assert classifier.save_ml(path)

        fresh = TicketClassifier()
        assert fresh.load_ml(path)
        result = fresh.classify("App crashes on startup")
        assert result["method"] == "tfidf-logreg"
