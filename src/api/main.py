# src/api/main.py
from fastapi import FastAPI
from .models import SequenceInput, PredictionResponse
from .utils import predict_rul

app = FastAPI(
    title="Automotive Predictive Maintenance API",
    description="Real-time RUL prediction API using LSTM sequence model",
    version="1.0"
)

@app.get("/")
def root():
    return {"message": "Predictive Maintenance API is running"}

@app.post("/predict", response_model=PredictionResponse)
def predict_rul_api(data: SequenceInput):
    pred = predict_rul(data.sequence)
    return PredictionResponse(
        predicted_rul=pred,
        used_sequence_length=len(data.sequence)
    )
