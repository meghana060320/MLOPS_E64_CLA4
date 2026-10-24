"""Python client to test the FastAPI prediction endpoint."""
import requests

BASE_URL = "http://127.0.0.1:8000"
payload = {
    "Pclass": 3, "Sex": "female", "Age": 24,
    "SibSp": 0, "Parch": 0, "Fare": 8.05, "Embarked": "S"
}

if __name__ == "__main__":
    print("Health:", requests.get(f"{BASE_URL}/health", timeout=10).json())
    response = requests.post(f"{BASE_URL}/predict", json=payload, timeout=10)
    print("Status:", response.status_code)
    print("Prediction:", response.json())
