"""Tests for the real-dataset loader (NLTK movie_reviews with fallback)."""

import pandas as pd

from src.data_loading import load_nltk_movie_reviews


def test_movie_reviews_or_fallback():
    """Returns a real corpus when available, otherwise falls back to sample data."""
    df = load_nltk_movie_reviews()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["text", "label"]
    assert len(df) > 0
    assert set(df["label"].unique()).issubset({0, 1})
    # Real corpus has ~2000 rows; sample fallback has 8
    assert len(df) in (8, 2000)


def test_movie_reviews_text_is_string():
    df = load_nltk_movie_reviews()
    assert df["text"].map(lambda t: isinstance(t, str)).all()
