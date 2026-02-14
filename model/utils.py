import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score,
    f1_score, matthews_corrcoef
)

def load_and_preprocess(csv_path, target_column):
    # Load dataset
    df = pd.read_csv(csv_path)

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X = pd.get_dummies(X, drop_first=True)

    # Save feature schema
    feature_columns = X.columns

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.15,
        random_state=42,
        stratify=y
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        feature_columns,
        scaler
    )


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

    try:
        if len(np.unique(y_test)) == 2:
            # Binary classification
            y_prob = model.predict_proba(X_test)[:, 1]
            metrics["AUC"] = roc_auc_score(y_test, y_prob)
        else:
            # Multi-class classification
            y_prob = model.predict_proba(X_test)
            metrics["AUC"] = roc_auc_score(
                y_test,
                y_prob,
                multi_class="ovr",
                average="weighted"
            )
    except Exception:
        metrics["AUC"] = "Not Applicable"

    return metrics
