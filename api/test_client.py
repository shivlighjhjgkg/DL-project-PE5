"""
Test client for FinBERT-LSTM Trading API
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_api():
    print("1. Testing /health endpoint...")
    try:
        r = requests.get(f"{BASE_URL}/health")
        print("Status code:", r.status_code)
        print("Response:", r.json())
    except Exception as e:
        print("Failed to connect to API:", e)
        return

    print("\n2. Testing /model/info endpoint...")
    r = requests.get(f"{BASE_URL}/model/info")
    print("Response:", r.json())

    print("\n3. Testing /predict endpoint (Ensemble)...")
    payload = {
        "prices": [12100.0, 12150.0, 12200.0, 12180.0, 12250.0, 12300.0, 12320.0, 12380.0, 12420.0, 12450.0],
        "sentiments": [0.1, 0.2, 0.05, -0.1, 0.3, 0.15, 0.4, 0.25, 0.3, 0.35],
        "model": "ensemble"
    }
    r = requests.post(f"{BASE_URL}/predict", json=payload)
    print("Status code:", r.status_code)
    print("Prediction:", json.dumps(r.json(), indent=2))

if __name__ == "__main__":
    test_api()
