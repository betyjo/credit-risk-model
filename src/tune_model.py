import pandas as pd

from sklearn.model_selection import (
    GridSearchCV,
    train_test_split
)

from sklearn.ensemble import (
    RandomForestClassifier
)

df = pd.read_csv(
    "data/processed/processed_data.csv"
)

X = df.drop(
    columns=[
        "CustomerId",
        "cluster",
        "is_high_risk"
    ]
)

y = df["is_high_risk"]

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
)

params = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None]
}

grid = GridSearchCV(
    RandomForestClassifier(
        random_state=42
    ),
    params,
    cv=3,
    scoring="roc_auc"
)

grid.fit(
    X_train,
    y_train
)

print(
    "Best Parameters:",
    grid.best_params_
)