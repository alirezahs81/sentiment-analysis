# 📊 Sentiment Analysis & Support Ticket Classification

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow)](https://huggingface.co/docs/transformers)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end NLP project combining **BERT-based sentiment analysis** with a
**rule-based support ticket classifier**, served through an interactive
Streamlit dashboard.

## ✨ Features

### Sentiment Analysis (Deep Learning)
- Fine-tunable `bert-base-uncased` classifier (PyTorch + 🤗 Transformers)
- Custom `Dataset` / training loop with linear warmup scheduling
- Single-text and batch prediction with confidence scores and class probabilities
- Model persistence (`models/sentiment_model.pt`) and JSON evaluation reports

### Ticket Classification (Rule-Based)
- Routes support tickets into 6 categories: Technical Issues, Billing & Payments,
  Account Access, Feature Requests, General Inquiries, Urgent/Priority
- Keyword-driven category scoring with confidence estimates
- Automatic urgency/priority estimation (low / medium / high)

### Interactive Dashboard (Streamlit)
- 📝 **Single Analysis** — real-time sentiment prediction with probability bars
- 📊 **Batch Analysis** — upload a CSV, get predictions, metrics and a downloadable results file
- 📈 **Model Evaluation** — accuracy, per-class metrics and confusion matrix
- 📁 **Data Management** — configurable preprocessing pipeline (cleaning, stopwords, stemming, lemmatization)
- 🎫 **Ticket Classification** — category, priority and keyword breakdown per ticket
- 🏋️ **Train Model** — kick off training directly from the UI and watch loss/accuracy curves

## 🗂 Project Structure

```
sentiment-analysis/
├── app.py                     # Streamlit dashboard (entry point)
├── src/
│   ├── data_loading.py        # Sample data + CSV I/O helpers
│   ├── preprocessing.py       # Text cleaning, stopwords, stemming, lemmatization
│   ├── model_training.py      # BERT classifier, Dataset and Trainer classes
│   ├── train.py               # Full training pipeline entry point
│   ├── evaluation.py          # Metrics, confusion-matrix plotting, report export
│   ├── inference.py           # SentimentAnalyzer for single/batch inference
│   └── ticket_classification.py  # Rule-based ticket classifier
├── tests/                     # Pytest suite
├── data/                      # Datasets (gitkept)
├── models/                    # Trained weights (gitignored)
├── reports/                   # Generated metrics and plots (gitignored)
└── requirements.txt
```

## 🚀 Quick Start

### 1. Install

```bash
git clone https://github.com/alirezahs81/sentiment-analysis.git
cd sentiment-analysis
python -m venv venv
venv\Scripts\activate        # Windows  (or: source venv/bin/activate)
pip install -r requirements.txt
```

### 2. Run the dashboard

```bash
streamlit run app.py
```

### 3. Train the model (optional)

```bash
python -m src.train
```

Trained weights are saved to `models/sentiment_model.pt`; metrics and training
history land in `reports/`. Without a trained model the app warns and uses
random weights — train once for meaningful predictions.

## 🧪 Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## 📌 Notes & Limitations

- The shipped sample dataset is tiny (8 examples) — it demonstrates the
  pipeline end to end. Plug in a real dataset (e.g. SST-2, IMDB or your own
  CSV with `text` and `label` columns) via `src/data_loading.py`.
- Ticket classification is intentionally rule-based for transparency and
  zero training cost; the `TicketDataset` class is included as a scaffold for
  upgrading it to an ML model.
- Persian text cleaning is supported by the preprocessor for future extension.

## 📄 License

Released under the [MIT License](LICENSE).
