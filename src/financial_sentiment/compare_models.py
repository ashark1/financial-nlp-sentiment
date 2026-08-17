from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from .evaluate import classification_metrics
from .finbert import predict_finbert
from .real_data import load_financial_phrasebank


def compare(model_path: str, limit: int | None = None) -> dict:
    frame = load_financial_phrasebank()
    if limit:
        frame = frame.head(limit)

    baseline = joblib.load(model_path)
    baseline_pred = baseline.predict(frame["text"])
    finbert_pred = [predict_finbert(text)["label"] for text in frame["text"]]

    return {
        "rows": int(len(frame)),
        "baseline": classification_metrics(frame["label"], baseline_pred),
        "finbert": classification_metrics(frame["label"], finbert_pred),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline and FinBERT on Financial PhraseBank.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default="artifacts/model_comparison.json")
    args = parser.parse_args()

    result = compare(args.model, args.limit)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
