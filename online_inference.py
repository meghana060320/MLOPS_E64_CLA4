"""Run one online-style prediction using a single unseen passenger record."""
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.pkl"
FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("model.pkl missing. Run: python train_model.py")
    model = joblib.load(MODEL_PATH)
    passenger = {
        "Pclass": 3, "Sex": "female", "Age": 24.0,
        "SibSp": 0, "Parch": 0, "Fare": 8.05, "Embarked": "S"
    }
    result = int(model.predict(pd.DataFrame([passenger], columns=FEATURES))[0])
    print("Online inference - one passenger")
    print(f"Input: {passenger}")
    print(f"Prediction: {'Survived (1)' if result == 1 else 'Did not survive (0)'}")

if __name__ == "__main__":
    main()
