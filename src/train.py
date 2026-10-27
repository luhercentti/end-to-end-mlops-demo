"""
Training pipeline for the Student Academic Risk prediction model.

Run:
    python src/train.py

Produces (all artifacts committed to disk, none to git):
    model.pkl            - trained RandomForestClassifier pipeline artifact
    scaler.pkl           - fitted StandardScaler
    label_encoder.pkl    - fitted LabelEncoder for the target classes
    feature_columns.pkl  - the exact column order expected at inference time
    plots/results.png    - confusion matrix used by the CML CI/CD report
"""
import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "data.csv")
ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "..")
PLOTS_DIR = os.path.join(ARTIFACT_DIR, "plots")


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    return df


def preprocess(df: pd.DataFrame):
    df = df.drop(columns=["id"], errors="ignore")

    X = df.drop(columns=["Target"])
    y = df["Target"]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    X = pd.get_dummies(X, columns=["Course"], drop_first=True)
    feature_columns = X.columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, label_encoder, feature_columns


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = load_data()
    X_scaled, y, scaler, label_encoder, feature_columns = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)

    mlflow.set_experiment("Student_Risk_Prediction")

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
    }

    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("Best params:", grid.best_params_)
    print("Accuracy:", acc)
    print(report)

    with mlflow.start_run():
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("best_cv_accuracy", grid.best_score_)
        mlflow.log_metric("test_accuracy", acc)
        mlflow.sklearn.log_model(best_model, "model")

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title(f"Confusion Matrix (accuracy={acc:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "results.png"))

    with open(os.path.join(ARTIFACT_DIR, "report.md"), "w") as f:
        f.write("# Student Risk Prediction Results\n\n")
        f.write(f"**Test accuracy:** {acc:.4f}\n\n")
        f.write("```\n" + report + "\n```\n")
        f.write("\n![confusion matrix](./plots/results.png)\n")

    joblib.dump(best_model, os.path.join(ARTIFACT_DIR, "model.pkl"))
    joblib.dump(scaler, os.path.join(ARTIFACT_DIR, "scaler.pkl"))
    joblib.dump(label_encoder, os.path.join(ARTIFACT_DIR, "label_encoder.pkl"))
    joblib.dump(feature_columns, os.path.join(ARTIFACT_DIR, "feature_columns.pkl"))

    print("Artifacts saved: model.pkl, scaler.pkl, label_encoder.pkl, feature_columns.pkl")


if __name__ == "__main__":
    main()
