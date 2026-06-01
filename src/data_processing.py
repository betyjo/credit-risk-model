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
def extract_time_features(df):

    df = df.copy()

    df["TransactionStartTime"] = pd.to_datetime(
        df["TransactionStartTime"]
    )

    df["hour"] = (
        df["TransactionStartTime"]
        .dt.hour
    )

    df["day"] = (
        df["TransactionStartTime"]
        .dt.day
    )

    df["month"] = (
        df["TransactionStartTime"]
        .dt.month
    )

    df["year"] = (
        df["TransactionStartTime"]
        .dt.year
    )

    df["weekday"] = (
        df["TransactionStartTime"]
        .dt.weekday
    )

    return df
def merge_customer_features(df):

    customer_features = (
        create_aggregate_features(df)
    )

    merged = df.merge(
        customer_features,
        on="CustomerId",
        how="left"
    )

    return merged
def build_preprocessor(
    numerical_cols,
    categorical_cols
):
        numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )
        categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )
        preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_pipeline,
                numerical_cols
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_cols
            )
        ]
    )
        return preprocessor
