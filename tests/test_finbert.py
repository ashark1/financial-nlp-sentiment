import pytest

from financial_sentiment.finbert import normalise_label, predict_finbert


def test_normalise_known_labels():
    assert normalise_label("positive") == "positive"
    assert normalise_label("LABEL_1") == "negative"
    assert normalise_label("label_2") == "neutral"


def test_normalise_rejects_unknown_label():
    with pytest.raises(ValueError, match="Unexpected FinBERT label"):
        normalise_label("unknown")


def test_predict_finbert_rejects_empty_text():
    with pytest.raises(ValueError, match="must not be empty"):
        predict_finbert("   ")
