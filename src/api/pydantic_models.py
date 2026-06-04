from pydantic import BaseModel


class CreditRequest(BaseModel):
    total_amount: float
    avg_amount: float
    std_amount: float
    min_amount: float
    max_amount: float
    transaction_count: int
    Recency: float
    Frequency: float
    Monetary: float


class CreditResponse(BaseModel):
    risk_probability: float