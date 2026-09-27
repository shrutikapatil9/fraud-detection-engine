import os
import time
from typing import Dict, Any
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field


ml_models: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
 
    print("Initializing microservice: Loading ML models and graph dictionaries...")
    try:
        ml_models["model"] = joblib.load("models/fraud_model.pkl")
        ml_models["scaler_amount"] = joblib.load("models/scaler_amount.pkl")
        ml_models["scaler_time"] = joblib.load("models/scaler_time.pkl")
        ml_models["graph_lookup"] = joblib.load("models/graph_lookup.pkl")
        ml_models["feature_names"] = joblib.load("models/feature_names.pkl")
        print("Models successfully loaded into memory.")
    except Exception as e:
        print(f"Error loading model artifacts: {e}")
        raise RuntimeError("Model artifacts missing. Run src/train.py first.")
    
    yield
    
    ml_models.clear()
    print("Microservice shutdown: Memory cleared.")


app = FastAPI(
    title="Real-Time Fraud Detection Engine",
    description="Low-latency microservice for scoring financial transactions.",
    version="1.0.0",
    lifespan=lifespan
)


class TransactionRequest(BaseModel):
 
    time: float = Field(..., description="Seconds elapsed since first transaction in dataset", example=400.0)
    amount: float = Field(..., description="Transaction amount in local currency", example=150.50)
    card_id: str = Field(..., description="Unique Identifier for Paying Card", example="card_1200")
    merchant_id: str = Field(..., description="Unique Identifier for Receiving Merchant", example="merchant_450")
    pca_features: Dict[str, float] = Field(
        ..., 
        description="Dictionary containing anonymized PCA features V1 through V28",
        example={f"V{i}": 0.01 for i in range(1, 29)}
    )


class FraudPredictionResponse(BaseModel):
   
    is_fraud: bool
    fraud_probability: float
    decision_threshold: float
    latency_ms: float


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
   
    return {"status": "healthy", "models_loaded": "model" in ml_models}


@app.post("/predict", response_model=FraudPredictionResponse)
def predict_fraud(transaction: TransactionRequest):
   
    start_time = time.perf_counter()

    try:
      
        scaled_amount = ml_models["scaler_amount"].transform([[transaction.amount]])[0][0]
        scaled_time = ml_models["scaler_time"].transform([[transaction.time]])[0][0]

       
        pagerank_dict = ml_models["graph_lookup"]["pagerank"]
        degree_dict = ml_models["graph_lookup"]["degree_centrality"]

        card_pr = pagerank_dict.get(transaction.card_id, 0.0)
        merchant_pr = pagerank_dict.get(transaction.merchant_id, 0.0)
        card_dc = degree_dict.get(transaction.card_id, 0.0)
        merchant_dc = degree_dict.get(transaction.merchant_id, 0.0)

       
        input_data = {}
        
       
        for i in range(1, 29):
            col_name = f"V{i}"
            input_data[col_name] = transaction.pca_features.get(col_name, 0.0)

      
        input_data["scaled_amount"] = scaled_amount
        input_data["scaled_time"] = scaled_time
        input_data["card_pagerank"] = card_pr
        input_data["merchant_pagerank"] = merchant_pr
        input_data["card_degree_centrality"] = card_dc
        input_data["merchant_degree_centrality"] = merchant_dc

       
        df_input = pd.DataFrame([input_data])[ml_models["feature_names"]]


        fraud_prob = float(ml_models["model"].predict_proba(df_input)[0][1])
        threshold = 0.50
        is_fraud = fraud_prob >= threshold

     
        elapsed_latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return FraudPredictionResponse(
            is_fraud=is_fraud,
            fraud_probability=round(fraud_prob, 4),
            decision_threshold=threshold,
            latency_ms=elapsed_latency_ms
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failure: {str(e)}"
        )