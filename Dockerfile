FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

COPY requirements.txt requirements-dev.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && python -c "import nltk; nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True)"

COPY . .

# Train lightweight ticket classifier so the app has a real model
RUN python -m src.train_tickets

EXPOSE 8501 8000

# Default: Streamlit dashboard. Override for API:
#   docker run ... uvicorn src.api:app --host 0.0.0.0 --port 8000
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
