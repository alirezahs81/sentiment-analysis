# Sentiment Analysis & Support Ticket Classification

[![CI](https://github.com/alirezahs81/sentiment-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/alirezahs81/sentiment-analysis/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow)](https://huggingface.co/docs/transformers)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

End-to-end NLP project combining **BERT-based sentiment analysis** with an
**ML ticket classifier** (TF-IDF + LogisticRegression, rule-based fallback),
served through a **Streamlit** dashboard and a **FastAPI** service.

## Features

### Sentiment analysis (deep learning)
- Fine-tunable `bert-base-uncased` classifier (PyTorch + 🤗 Transformers)
- Single-text and batch prediction with confidence and class probabilities
- Model persistence (`models/sentiment_model.pt`) and JSON evaluation reports

### Ticket classification (ML + rule fallback)
- **Primary:** TF-IDF (word bigrams) + LogisticRegression on bundled labeled tickets
- **Fallback:** transparent keyword rules when no trained model is present
- Categories: Technical Issues, Billing & Payments, Account Access, Feature Requests, General Inquiries
- Automatic priority estimation (low / medium / high)

| Ticket classifier | 5-fold CV accuracy |
|---|---|
| Rule-based keywords | ~0.19 |
| TF-IDF + LogisticRegression | ~0.69 |

The labeled ticket set is small and hand-written; numbers show direction, not production quality.

### Serving
- **Streamlit** interactive dashboard (analysis, batch CSV, evaluation, training hooks)
- **FastAPI** REST endpoints for tickets (always) and sentiment (when BERT weights exist)
- **Docker** image trains the ticket model at build time

## Project structure

```text
sentiment-analysis/
├── app.py                     # Streamlit dashboard
├── src/
│   ├── data_loading.py
│   ├── preprocessing.py
│   ├── model_training.py      # BERT classifier + trainer
│   ├── train.py               # Sentiment training CLI
│   ├── evaluation.py
│   ├── inference.py           # SentimentAnalyzer
│   ├── ticket_classification.py
│   ├── train_tickets.py       # Ticket training CLI
│   └── api.py                 # FastAPI app
├── tests/
├── data/tickets.csv
├── Dockerfile
└── requirements.txt
```

## Quick start

```bash
git clone https://github.com/alirezahs81/sentiment-analysis.git
cd sentiment-analysis
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Optional: ticket model for better routing
python -m src.train_tickets

# Dashboard
streamlit run app.py

# API (docs at http://localhost:8000/docs)
uvicorn src.api:app --reload --port 8000
```

### Sentiment training (optional, GPU recommended)

```bash
python -m src.train --dataset movie_reviews
```

Without a trained BERT checkpoint the UI warns and sentiment weights are random.
Ticket endpoints still work after `python -m src.train_tickets`.

### Docker

```bash
docker build -t sentiment-analysis .
docker run --rm -p 8501:8501 sentiment-analysis
# API instead of Streamlit:
docker run --rm -p 8000:8000 sentiment-analysis \
  uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### Example API call

```bash
curl -s http://localhost:8000/tickets/classify \
  -H 'Content-Type: application/json' \
  -d '{"text": "Payment charged twice, need a refund"}'
```

## Tests & CI

```bash
pip install -r requirements-dev.txt
pytest
```

GitHub Actions runs on **Python 3.10, 3.11, and 3.12** (lightweight stack: sklearn/NLTK/FastAPI — not full torch fine-tuning).

## Notes & limitations

- Fine-tuning `bert-base-uncased` on the full movie-review corpus needs a GPU for reasonable time; on CPU use small `--epochs` or subsample.
- BERT weights and ticket `.joblib` artifacts are not committed; train locally or via Docker build for tickets.
- CI covers preprocessing, data loading, evaluation, ticket classifiers, and API ticket routes — not GPU BERT training.

## License

[MIT License](LICENSE).
