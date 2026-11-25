"""Tests for the text preprocessing pipeline."""

from src.preprocessing import TextPreprocessor


def test_clean_text_removes_urls_and_mentions():
    p = TextPreprocessor(language="english")
    text = "Check https://example.com and @user #tag NOW!!!"
    cleaned = p.clean_text(text)
    assert "http" not in cleaned
    assert "@user" not in cleaned
    assert "#tag" not in cleaned
    assert cleaned == cleaned.lower()


def test_clean_text_handles_non_string():
    p = TextPreprocessor(language="english")
    assert p.clean_text(None) == ""
    assert p.clean_text(123) == ""


def test_remove_stopwords():
    p = TextPreprocessor(language="english")
    result = p.remove_stopwords("this is a great product")
    assert "this" not in result.split()
    assert "great" in result.split()


def test_preprocess_full_pipeline():
    p = TextPreprocessor(language="english")
    result = p.preprocess("I LOVE this product! It's amazing!!!")
    assert isinstance(result, str)
    assert result == result.lower()
    assert "!" not in result


def test_persian_cleaning_keeps_persian_characters():
    p = TextPreprocessor(language="persian")
    cleaned = p.clean_text("این محصول عالی است!")
    assert "این" in cleaned
    assert "عالی" in cleaned
