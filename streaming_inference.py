"""Simulate streaming inference by processing 10 records one by one."""
from pathlib import Path
import time
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "train.csv"
MODEL_PATH = ROOT / "model.pkl"
FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Missing data/train.csv; see README.md.")
    if not MODEL_PATH.exists():
        raise FileNotFoundError("model.pkl missing. Run: python train_model.py")
    df = pd.read_csv(DATA_PATH)
    records = df.drop(columns=["Survived"]).sample(n=10, random_state=202).reset_index(drop=True)
    model = joblib.load(MODEL_PATH)
    print("Streaming inference - one incoming passenger per second")
    for i, row in records.iterrows():
        record = row[FEATURES].to_frame().T
        prediction = int(model.predict(record)[0])
        print(f"Record {i+1:02d}: PassengerId={row.get('PassengerId', 'N/A')} -> "
              f"{'Survived (1)' if prediction else 'Did not survive (0)'}")
        if i < len(records) - 1:
            time.sleep(1)

if __name__ == "__main__":
    main()
