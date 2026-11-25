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

### Ticket Classification (ML with rule-based fallback)
- TF-IDF (word bigrams) + LogisticRegression trained on a bundled set of 58 labeled tickets
- Routes support tickets into 5 categories: Technical Issues, Billing & Payments,
  Account Access, Feature Requests, General Inquiries
- Transparent rule-based keyword classifier is used automatically when no trained model exists
- Automatic urgency/priority estimation (low / medium / high)

| Ticket classifier | 5-fold CV accuracy |
|---|---|
| Rule-based keywords | 0.19 |
| TF-IDF + LogisticRegression | 0.69 |

The labeled ticket set is small and hand-written, so these numbers indicate
direction rather than production quality — adding real tickets improves it quickly.

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
│   ├── ticket_classification.py  # ML ticket classifier + rule-based fallback
│   └── train_tickets.py       # Ticket classifier training entry point
├── tests/                     # Pytest suite
├── data/                      # tickets.csv (labeled ticket dataset)
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

`--dataset sample` (default) uses the bundled 8-example demo set. To train on a
real corpus (~2,000 labeled movie reviews, downloaded automatically on first use):

```bash
python -m src.train --dataset movie_reviews
```

### 4. Train the ticket classifier (optional, recommended)

```bash
python -m src.train_tickets
```

Trains the ML ticket classifier on `data/tickets.csv` and saves it to
`models/ticket_classifier.joblib`; the app loads it automatically on startup.
You can point it at your own CSV (`--data path.csv`) with `ticket_text` and
`category` columns.

## 🧪 Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## 📌 Notes & Limitations

- Fine-tuning `bert-base-uncased` on the full movie-review corpus needs a GPU
  to finish in reasonable time; on CPU use a small `--epochs` value or subsample.
- The BERT model and the trained ticket classifier are not committed (binary
  artifacts); run the training commands above to produce them locally.
- The Streamlit app and BERT training require `torch`, which is not installed in
  the CI environment — CI covers preprocessing, data loading, evaluation and the
  ticket classifiers only.
- Persian text cleaning is supported by the preprocessor for future extension.

## 📄 License

Released under the [MIT License](LICENSE).
