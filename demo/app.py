
# Credit Card Fraud Detection — FastAPI REST Service

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import numpy as np
import pandas as pd
import joblib
import time
import os

# Load Model Artifact
ARTIFACT_PATH = os.getenv("MODEL_PATH", "../deploy/fraud_model_artifact.pkl")

artifact      = joblib.load(ARTIFACT_PATH)
MODEL         = artifact["model"]
SCALER        = artifact["scaler"]
FEATURE_NAMES = artifact["feature_names"]
THRESHOLD     = artifact["threshold"]
METADATA      = artifact["metadata"]
N_FEATURES    = len(FEATURE_NAMES)

# FastAPI App
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Real-time fraud scoring using XGBoost",
    version=METADATA.get("version", "1.0.0")
)

# Input Schema
class TransactionFeatures(BaseModel):
    Time: float = Field(..., ge=0)
    Amount: float = Field(..., ge=0)

    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float


class PredictionResponse(BaseModel):
    fraud_probability: float
    fraud_predicted: bool
    risk_level: str
    threshold_used: float
    inference_ms: float
    model_version: str


class BatchRequest(BaseModel):
    transactions: List[TransactionFeatures]


class BatchResponse(BaseModel):
    results: List[PredictionResponse]
    total_transactions: int
    fraud_flagged: int
    total_ms: float
    avg_ms_per_tx: float


# Risk Level Function
def risk_level(prob):
    if prob < 0.30:
        return "LOW"
    elif prob < 0.70:
        return "MEDIUM"
    else:
        return "HIGH"


# Preprocess Function
def preprocess(tx: TransactionFeatures):
    hour = (tx.Time / 3600) % 24
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    log_amount = np.log1p(tx.Amount)

    v_features = [getattr(tx, f"V{i}") for i in range(1, 29)]

    raw_vec = np.array(
        v_features + [tx.Amount, log_amount, hour_sin, hour_cos]
    ).reshape(1, -1)

    scaled = SCALER.transform(raw_vec)

    return scaled


# Health Check
@app.get("/")
def health():
    return {
        "status": "healthy",
        "service": "Fraud Detection API",
        "version": METADATA.get("version")
    }


# Model Info
@app.get("/model/info")
def model_info():
    return {
        "model_metadata": METADATA,
        "n_features": N_FEATURES,
        "feature_names": FEATURE_NAMES,
        "decision_threshold": THRESHOLD
    }


# Single Prediction
@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionFeatures):

    t0 = time.time()

    try:
        X = preprocess(transaction)
        prob = float(MODEL.predict_proba(X)[0, 1])
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    elapsed = (time.time() - t0) * 1000

    return PredictionResponse(
        fraud_probability=round(prob, 6),
        fraud_predicted=prob >= THRESHOLD,
        risk_level=risk_level(prob),
        threshold_used=THRESHOLD,
        inference_ms=round(elapsed, 3),
        model_version=METADATA.get("version", "1.0.0")
    )


# Batch Prediction
@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest):

    t0 = time.time()

    results = []

    for tx in request.transactions:

        t_tx = time.time()

        X = preprocess(tx)
        prob = float(MODEL.predict_proba(X)[0, 1])

        ms = (time.time() - t_tx) * 1000

        results.append(
            PredictionResponse(
                fraud_probability=round(prob, 6),
                fraud_predicted=prob >= THRESHOLD,
                risk_level=risk_level(prob),
                threshold_used=THRESHOLD,
                inference_ms=round(ms, 3),
                model_version=METADATA.get("version", "1.0.0")
            )
        )

    total_ms = (time.time() - t0) * 1000
    flagged = sum(r.fraud_predicted for r in results)

    return BatchResponse(
        results=results,
        total_transactions=len(results),
        fraud_flagged=flagged,
        total_ms=round(total_ms, 2),
        avg_ms_per_tx=round(total_ms / len(results), 3)
    )
