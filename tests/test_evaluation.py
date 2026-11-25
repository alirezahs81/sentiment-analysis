"""Tests for evaluation utilities."""

from src.evaluation import evaluate_model, classification_report_df


def test_evaluate_model_perfect_predictions():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 1]
    results = evaluate_model(y_true, y_pred, ["Negative", "Positive"])
    assert results["overall"]["accuracy"] == 1.0
    assert results["overall"]["f1_macro"] == 1.0
    assert set(results["per_class"]) == {"Negative", "Positive"}


def test_evaluate_model_report_keys():
    y_true = [0, 1, 0, 1]
    y_pred = [1, 1, 0, 0]
    results = evaluate_model(y_true, y_pred, ["Negative", "Positive"])
    assert 0.0 <= results["overall"]["accuracy"] <= 1.0
    assert "Negative" in results["report"]


def test_classification_report_df():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 1]
    results = evaluate_model(y_true, y_pred, ["Negative", "Positive"])
    df = classification_report_df(results["report"])
    assert "precision" in df.columns
