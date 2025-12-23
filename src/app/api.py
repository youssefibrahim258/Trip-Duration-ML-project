"""
FastAPI application for Trip Duration prediction.

This module exposes a REST endpoint that receives trip information
and returns the predicted trip duration using a trained ML model.
"""

from fastapi import FastAPI
from pydantic import BaseModel

from src.serving.inference import predict

app = FastAPI(title="Trip Duration API")


class TripInput(BaseModel):
    id: str
    vendor_id: int
    pickup_datetime: str
    passenger_count: int
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    store_and_fwd_flag: str


@app.post("/predict")
def predict_trip(data: TripInput):
    result = predict(data.dict())
    return {"trip_duration": result}
