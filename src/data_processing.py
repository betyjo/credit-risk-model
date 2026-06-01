import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.impute import SimpleImputer
def create_aggregate_features(df):

    agg_df = (
        df.groupby("CustomerId")
        .agg(
            total_amount=("Amount", "sum"),
            avg_amount=("Amount", "mean"),
            std_amount=("Amount", "std"),
            min_amount=("Amount", "min"),
            max_amount=("Amount", "max"),
            transaction_count=("TransactionId", "count")
        )
        .reset_index()
    )

    return agg_df