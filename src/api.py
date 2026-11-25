"""FastAPI inference service.

Ticket classification always available (sklearn).
Sentiment endpoints require torch + transformers and a trained model file.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.ticket_classification import TicketClassifier, train_ticket_classifier

app = FastAPI(
    title="Sentiment Analysis & Ticket Classification API",
    description=(
        "Ticket routing (TF-IDF + LogisticRegression with rule fallback) "
        "and optional BERT sentiment analysis."
    ),
    version="1.0.0",
)

_ticket_clf: TicketClassifier | None = None
_sentiment = None
_sentiment_error: str | None = None


def get_ticket_classifier() -> TicketClassifier:
    global _ticket_clf
    if _ticket_clf is None:
        clf = TicketClassifier()
        if not clf.load_ml():
            # Train from bundled data so the API works out of the box
            clf = train_ticket_classifier(save=True)
        _ticket_clf = clf
    return _ticket_clf


def get_sentiment_analyzer():
    """Lazy-load BERT analyzer; may be unavailable in lightweight installs."""
    global _sentiment, _sentiment_error
    if _sentiment is not None:
        return _sentiment
    if _sentiment_error is not None:
        raise HTTPException(status_code=503, detail=_sentiment_error)
    try:
        from src.inference import SentimentAnalyzer

        _sentiment = SentimentAnalyzer()
        return _sentiment
    except Exception as exc:  # noqa: BLE001 — surface install/runtime issues
        _sentiment_error = (
            f"Sentiment model unavailable: {exc}. "
            "Install torch/transformers and train with: python -m src.train"
        )
        raise HTTPException(status_code=503, detail=_sentiment_error) from exc


class TextIn(BaseModel):
    text: str = Field(..., min_length=1, examples=["App crashes on startup"])


class BatchIn(BaseModel):
    texts: list[str] = Field(..., min_length=1)


@app.get("/health")
def health() -> dict[str, Any]:
    tickets_ready = True
    try:
        get_ticket_classifier()
    except Exception:  # noqa: BLE001
        tickets_ready = False

    sentiment_ready = False
    try:
        from pathlib import Path

        from src.inference import MODEL_PATH

        sentiment_ready = Path(MODEL_PATH).exists()
    except Exception:  # noqa: BLE001
        sentiment_ready = False

    return {
        "status": "ok",
        "tickets_ready": tickets_ready,
        "sentiment_model_on_disk": sentiment_ready,
    }


@app.post("/tickets/classify")
def classify_ticket(body: TextIn) -> dict[str, Any]:
    clf = get_ticket_classifier()
    return clf.classify(body.text)


@app.post("/tickets/classify/batch")
def classify_tickets_batch(body: BatchIn) -> dict[str, Any]:
    clf = get_ticket_classifier()
    return {"predictions": [clf.classify(t) for t in body.texts]}


@app.post("/sentiment")
def analyze_sentiment(body: TextIn) -> dict[str, Any]:
    analyzer = get_sentiment_analyzer()
    return analyzer.analyze(body.text)


@app.post("/sentiment/batch")
def analyze_sentiment_batch(body: BatchIn) -> dict[str, Any]:
    analyzer = get_sentiment_analyzer()
    return {"predictions": analyzer.batch_analyze(body.texts)}
