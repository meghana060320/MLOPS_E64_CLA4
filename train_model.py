"""Train the Titanic classifier and save the complete preprocessing/model pipeline."""
from pathlib import Path
import json
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "train.csv"
MODEL_PATH = ROOT / "model.pkl"
METRICS_PATH = ROOT / "metrics.json"

FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
NUMERIC = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
CATEGORICAL = ["Sex", "Embarked"]

def build_pipeline():
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    prep = ColumnTransformer([
        ("num", numeric_pipe, NUMERIC),
        ("cat", categorical_pipe, CATEGORICAL),
    ])
    return Pipeline([
        ("preprocess", prep),
        ("classifier", RandomForestClassifier(
            n_estimators=250, random_state=42, class_weight="balanced"
        )),
    ])

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Missing data/train.csv. Download train.csv from "
            "https://www.kaggle.com/competitions/titanic/data and place it in the data folder."
        )
    df = pd.read_csv(DATA_PATH)
    required = set(FEATURES + ["Survived"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"train.csv is missing required columns: {sorted(missing)}")
    X = df[FEATURES]
    y = df["Survived"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    accuracy = float(accuracy_score(y_test, predictions))
    # Save the final pipeline trained on all labelled rows after evaluation.
    pipeline.fit(X, y)
    joblib.dump(pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps({
        "model": "Random Forest Classifier",
        "dataset": "Kaggle Titanic - Machine Learning from Disaster (train.csv)",
        "test_size": 0.2,
        "random_state": 42,
        "accuracy": accuracy,
        "training_rows": int(len(df)),
    }, indent=2))
    print(f"Model: Random Forest Classifier")
    print(f"Dataset rows: {len(df)}")
    print(f"Hold-out accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Saved pipeline: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")

if __name__ == "__main__":
    main()
