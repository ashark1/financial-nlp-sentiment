from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .finbert import predict_finbert

app = FastAPI(title="Financial NLP Sentiment API", version="0.2.0")


class PredictionRequest(BaseModel):
    text: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest) -> dict:
    try:
        return predict_finbert(request.text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
