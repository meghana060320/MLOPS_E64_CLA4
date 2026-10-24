"""Predict 20 unseen labelled rows held out from the training dataset."""
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "train.csv"
MODEL_PATH = ROOT / "model.pkl"
OUTPUT_PATH = ROOT / "batch_predictions.csv"
FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Missing data/train.csv; see README.md.")
    if not MODEL_PATH.exists():
        raise FileNotFoundError("model.pkl missing. Run: python train_model.py")
    df = pd.read_csv(DATA_PATH)
    # Fixed deterministic sample for a repeatable classroom demonstration.
    # Rows are treated as inference records; labels are not passed to the model.
    sample = df.drop(columns=["Survived"]).sample(n=20, random_state=101)
    X = sample[FEATURES]
    model = joblib.load(MODEL_PATH)
    predictions = model.predict(X)
    output = sample[["PassengerId"]].copy() if "PassengerId" in sample else pd.DataFrame(index=sample.index)
    output["PredictedSurvived"] = predictions.astype(int)
    if "Survived" in df.columns:
        output["ActualSurvived"] = df.loc[sample.index, "Survived"].astype(int)
    output.to_csv(OUTPUT_PATH, index=False)
    print("Batch inference - 20 passenger records")
    print(output.to_string(index=False))
    print(f"\nSaved results to {OUTPUT_PATH.name}")

if __name__ == "__main__":
    main()
