"""Tests for data loading utilities."""

import pandas as pd

from src.data_loading import load_sample_data, load_from_csv, save_processed_data


def test_load_sample_data():
    df = load_sample_data()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["text", "label"]
    assert len(df) > 0
    assert set(df["label"].unique()).issubset({0, 1})


def test_load_from_csv_missing_file_raises():
    import pytest

    with pytest.raises(FileNotFoundError):
        load_from_csv("does_not_exist.csv")


def test_save_and_load_roundtrip():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        df = load_sample_data()
        out = Path(tmp) / "processed.csv"
        save_processed_data(df, out)
        loaded = load_from_csv(out)
        assert len(loaded) == len(df)
