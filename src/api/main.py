from fastapi import FastAPI
from src.api.pydantic_models import (
    CreditRequest,
    CreditResponse
)

import pandas as pd
import joblib


app = FastAPI(
    title="Credit Risk API",
    version="1.0"
)


model = joblib.load("best_model.pkl")


@app.get("/")
def home():
    return {
        "message":
        "Credit Risk API Running"
    }


@app.post(
    "/predict",
    response_model=CreditResponse
)
def predict(data: CreditRequest):

    features = pd.DataFrame(
        [data.dict()]
    )

    probability = (
        model.predict_proba(features)
        [:, 1][0]
    )

    return CreditResponse(
        risk_probability=float(
            probability
        )
    )