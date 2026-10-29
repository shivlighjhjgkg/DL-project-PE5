"""
Automated Unit & Integration Tests for FinBERT-LSTM Trading API
==============================================================
Tests:
1. /health - Checks model availability and scaler
2. /model/info - Verifies all 4 architectures + ensemble are registered
3. /predict - Tests inference for each model architecture
4. /api/backtest/data - Validates simulation output structure and equity curves
"""

import sys
import os
import json

API_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "api")
sys.path.insert(0, API_DIR)

from fastapi.testclient import TestClient
from app import app, load_assets

client = TestClient(app)

def setup_module():
    load_assets()

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["scaler_loaded"] is True
    assert len(data["loaded_models"]) >= 3

def test_model_info():
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "available_models" in data
    assert "mlp" in data["available_models"]
    assert "lstm" in data["available_models"]
    assert "finbert_lstm" in data["available_models"]
    assert "attention_bilstm" in data["available_models"]

def test_prediction_all_models():
    sample_prices = [12100.0, 12150.0, 12200.0, 12180.0, 12250.0, 12300.0, 12320.0, 12380.0, 12420.0, 12450.0]
    sample_sent = [0.1, 0.2, 0.05, -0.1, 0.3, 0.15, 0.4, 0.25, 0.3, 0.35]

    for model_name in ["mlp", "lstm", "finbert_lstm", "attention_bilstm", "ensemble"]:
        payload = {
            "prices": sample_prices,
            "sentiments": sample_sent,
            "model": model_name
        }
        res = client.post("/predict", json=payload)
        assert res.status_code == 200, f"Model {model_name} failed: {res.text}"
        data = res.json()
        assert "predicted_price" in data
        assert "signal" in data
        assert data["signal"] in ["BUY", "SELL", "HOLD"]
        assert data["predicted_price"] > 5000.0  # reasonable NASDAQ price level
        print(f"[TEST PASSED] {model_name}: Predicted ${data['predicted_price']:.2f} ({data['signal']})")

def test_backtest_data_endpoint():
    res = client.get("/api/backtest/data?model=ensemble&scope=test")
    assert res.status_code == 200
    data = res.json()
    assert "dates" in data
    assert "actual_prices" in data
    assert "predicted_prices" in data
    assert "portfolio_history" in data
    assert "metrics" in data
    assert len(data["actual_prices"]) == len(data["predicted_prices"])
    assert data["metrics"]["initial_capital"] == 10000.0
    print(f"[TEST PASSED] Backtest test scope returned {len(data['dates'])} days.")

if __name__ == "__main__":
    setup_module()
    test_health()
    test_model_info()
    test_prediction_all_models()
    test_backtest_data_endpoint()
    print("\nALL API TESTS PASSED SUCCESSFULLY!")
