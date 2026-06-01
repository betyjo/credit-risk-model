import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
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
def prepare_dataset(df):

    df = extract_time_features(df)

    df = merge_customer_features(df)

    return df
if __name__ == "__main__":

    df = pd.read_csv(
        "data/raw/data.csv"
    )

    processed_df = prepare_dataset(df)

    processed_df.to_csv(
        "data/processed/processed_data.csv",
        index=False
    )

    print(
        "Processed dataset saved."
    )
def calculate_rfm(df):

    df = df.copy()

    df["TransactionStartTime"] = pd.to_datetime(
        df["TransactionStartTime"]
    )

    snapshot_date = (
        df["TransactionStartTime"].max()
        + pd.Timedelta(days=1)
    )

    rfm = (
        df.groupby("CustomerId")
        .agg(
            Recency=(
                "TransactionStartTime",
                lambda x:
                (
                    snapshot_date
                    - x.max()
                ).days
            ),

            Frequency=(
                "TransactionId",
                "count"
            ),

            Monetary=(
                "Amount",
                "sum"
            )
        )
        .reset_index()
    )

    return rfm
def scale_rfm(rfm):

    scaler = StandardScaler()

    scaled = scaler.fit_transform(
        rfm[
            [
                "Recency",
                "Frequency",
                "Monetary"
            ]
        ]
    )

    return scaled
def create_rfm_clusters(rfm):

    scaled_rfm = scale_rfm(rfm)

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    rfm["cluster"] = (
        kmeans.fit_predict(
            scaled_rfm
        )
    )

    return rfm