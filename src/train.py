import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

import mlflow
import mlflow.sklearn

from src.data_processing import prepare_dataset


# -----------------------------
# LOAD DATA
# -----------------------------
def load_data(path="data/raw/xente.csv"):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


# -----------------------------
# EVALUATION FUNCTION
# -----------------------------
def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs)
    }


# -----------------------------
# MAIN TRAINING PIPELINE
# -----------------------------
def main():

    mlflow.set_experiment("credit-risk-model")

    df = load_data()

    # Feature engineering pipeline
    df = prepare_dataset(df)

    # Target + features
    target = "is_high_risk"

    df = df.dropna(subset=[target])

    X = df.drop(columns=[target])
    y = df[target]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "decision_tree": DecisionTreeClassifier(max_depth=5),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    }

    best_model = None
    best_score = 0

    for name, model in models.items():

        with mlflow.start_run(run_name=name):

            model.fit(X_train, y_train)

            metrics = evaluate_model(model, X_test, y_test)

            # Log parameters
            mlflow.log_param("model_name", name)

            # Log metrics
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            # Log model
            mlflow.sklearn.log_model(model, name)

            print(f"\n{name}")
            print(metrics)

            # Track best model
            if metrics["roc_auc"] > best_score:
                best_score = metrics["roc_auc"]
                best_model = model

    print("\nBest Model ROC-AUC:", best_score)


if __name__ == "__main__":
    main()