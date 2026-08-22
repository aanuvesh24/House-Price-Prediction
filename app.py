# ===============================
# HOUSE PRICE PREDICTION API
# ===============================

import os
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

# -------------------------------
# Load model + feature order
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "house_price_model.pkl"))
features = joblib.load(os.path.join(BASE_DIR, "model_features.pkl"))

# -------------------------------
# FastAPI app
# -------------------------------
app = FastAPI(title="House Price Prediction API")

# -------------------------------
# Input Schema (request body)
# -------------------------------
class HouseData(BaseModel):
    area: float
    bedrooms: int
    bathrooms: int
    stories: int
    mainroad: str  # "yes" or "no"
    guestroom: str  # "yes" or "no"
    basement: str  # "yes" or "no"
    hotwaterheating: str  # "yes" or "no"
    airconditioning: str  # "yes" or "no"
    parking: int
    prefarea: str  # "yes" or "no"
    furnishingstatus: str  # "furnished", "semi-furnished", or "unfurnished"


# -------------------------------
# Home route
# -------------------------------
@app.get("/")
def home():
    return {"message": "House Price Prediction API Running 🚀"}


# -------------------------------
# Prediction route
# -------------------------------
@app.post("/predict")
def predict(data: HouseData):

    # Convert input → dataframe
    input_df = pd.DataFrame([data.model_dump() if hasattr(data, "model_dump") else data.dict()])

    # Ensure correct column order
    input_df = input_df[features]

    # Predict (log price)
    pred_log = model.predict(input_df)

    # Convert back from log
    pred_price = np.expm1(pred_log)

    return {
        "predicted_price": round(float(pred_price[0]), 2)
    }