import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

# -----------------------------------
# LOAD DATA
# -----------------------------------

df = pd.read_csv(
    "data/processed/processed_data.csv"
)

# -----------------------------------
# FEATURES / TARGET
# -----------------------------------

X = df.drop(
    columns=[
        "CustomerId",
        "cluster",
        "is_high_risk"
    ],
    errors="ignore"
)

y = df["is_high_risk"]

# -----------------------------------
# TRAIN TEST SPLIT
# -----------------------------------

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
)

# -----------------------------------
# MODELS
# -----------------------------------

models = {
    "LogisticRegression":
        LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

    "RandomForest":
        RandomForestClassifier(
            random_state=42
        ),

    "GradientBoosting":
        GradientBoostingClassifier(
            random_state=42
        )
}

# -----------------------------------
# MLFLOW EXPERIMENT
# -----------------------------------

mlflow.set_experiment(
    "credit-risk-model"
)

best_model = None
best_auc = 0

# -----------------------------------
# TRAIN MODELS
# -----------------------------------

for model_name, model in models.items():

    with mlflow.start_run(
        run_name=model_name
    ):

        model.fit(
            X_train,
            y_train
        )

        preds = model.predict(X_test)

        probs = model.predict_proba(
            X_test
        )[:, 1]

        accuracy = accuracy_score(
            y_test,
            preds
        )

        precision = precision_score(
            y_test,
            preds
        )

        recall = recall_score(
            y_test,
            preds
        )

        f1 = f1_score(
            y_test,
            preds
        )

        auc = roc_auc_score(
            y_test,
            probs
        )

        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        mlflow.log_metric(
            "precision",
            precision
        )

        mlflow.log_metric(
            "recall",
            recall
        )

        mlflow.log_metric(
            "f1_score",
            f1
        )

        mlflow.log_metric(
            "roc_auc",
            auc
        )

        mlflow.sklearn.log_model(
            model,
            model_name
        )

        print(f"\n{model_name}")
        print(f"AUC: {auc:.4f}")

        if auc > best_auc:

            best_auc = auc
            best_model = model

# -----------------------------------
# SAVE BEST MODEL
# -----------------------------------

joblib.dump(
    best_model,
    "best_model.pkl"
)

print(
    f"\nBest ROC-AUC: {best_auc:.4f}"
)

print(
    "Saved: best_model.pkl"
)