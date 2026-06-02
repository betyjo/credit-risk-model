from src.data_processing import (
    create_customer_features
)

import pandas as pd


def test_customer_features():

    sample = pd.DataFrame({
        "CustomerId": [1,1,2],
        "Amount": [100,200,300]
    })

    result = create_customer_features(
        sample
    )

    assert "total_amount" in result.columns


def test_customer_count():

    sample = pd.DataFrame({
        "CustomerId": [1,1,2],
        "Amount": [100,200,300]
    })

    result = create_customer_features(
        sample
    )

    assert len(result) == 2