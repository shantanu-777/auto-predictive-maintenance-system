# src/api/models.py
from pydantic import BaseModel
from typing import List

class SequenceInput(BaseModel):
    sequence: List[List[float]]   # shape: list of [features]

class PredictionResponse(BaseModel):
    predicted_rul: float
    used_sequence_length: int
class ErrorResponse(BaseModel):
    error_message: str