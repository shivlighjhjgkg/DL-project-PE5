"""
FastAPI Server for FinBERT-LSTM Stock Price Prediction and Trading Bot
Includes REST API + Interactive Animated Web Dashboard
======================================================================
Supports all four course models:
1. MLP Baseline (Priyanshu Singh)
2. Standard LSTM (Shivli Dimri)
3. FinBERT-LSTM Fusion (Dakshayani Sharma)
4. Attention-BiLSTM (Prajika Saini)
5. Blended Quantitative Ensemble
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import List, Optional

# Set local cache paths
API_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(API_DIR)
os.environ.setdefault("KERAS_HOME", os.path.join(PROJECT_ROOT, ".keras"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(PROJECT_ROOT, ".matplotlib"))
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import uvicorn

# Register Attention Layer for model loading
@keras.utils.register_keras_serializable()
class TemporalAttentionLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(TemporalAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        hidden_dim = input_shape[-1]
        self.W = self.add_weight(
            name="attention_weight",
            shape=(hidden_dim, hidden_dim),
            initializer="glorot_uniform",
            trainable=True
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(hidden_dim,),
            initializer="zeros",
            trainable=True
        )
        self.v = self.add_weight(
            name="context_vector",
            shape=(hidden_dim, 1),
            initializer="glorot_uniform",
            trainable=True
        )
        super(TemporalAttentionLayer, self).build(input_shape)

    def call(self, inputs):
        score = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        score = tf.tensordot(score, self.v, axes=[-1, 0])
        attention_weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(inputs * attention_weights, axis=1)
        return context, attention_weights

    def get_config(self):
        return super(TemporalAttentionLayer, self).get_config()


@keras.utils.register_keras_serializable()
class AttentionLayer(TemporalAttentionLayer):
    pass


app = FastAPI(
    title="FinBERT-LSTM Trading System API",
    description="Production API for NASDAQ Stock Price Prediction and Multi-Model Trading Simulation",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def find_file(local_name, root_subpath):
    p1 = os.path.join(API_DIR, local_name)
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(PROJECT_ROOT, *root_subpath.split("/"))
    if os.path.exists(p2):
        return p2
    return p1

SCALER_PATH = find_file("price_scaler.pkl", "models/price_scaler.pkl")
MLP_PATH = find_file("mlp_model.keras", "models/mlp_model.keras")
LSTM_PATH = find_file("lstm_model.keras", "models/lstm_model.keras")
FINBERT_LSTM_PATH = find_file("finbert_lstm_model.keras", "models/finbert_lstm_model.keras")
ATTN_BILSTM_PATH = find_file("attention_bilstm_model.keras", "models/attention_bilstm_model.keras")

STOCK_CSV_PATH = find_file("stock_price.csv", "data/raw/stock_price.csv")
SENTIMENT_CSV_PATH = find_file("sentiment_new.csv", "data/processed/sentiment_new.csv")
if not os.path.exists(SENTIMENT_CSV_PATH):
    SENTIMENT_CSV_PATH = find_file("sentiment.csv", "data/processed/sentiment.csv")

state = {
    "scaler": None,
    "models": {},
    "config": {
        "sequence_length": 10,
        "threshold_pct": 0.5
    },
    "cache": {}
}


def load_assets():
    print("Loading models and scaler...")
    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, "rb") as f:
            state["scaler"] = pickle.load(f)
        print("[OK] Price scaler loaded.")

    custom_objs = {
        'TemporalAttentionLayer': TemporalAttentionLayer,
        'AttentionLayer': AttentionLayer
    }

    model_files = {
        'mlp': MLP_PATH,
        'lstm': LSTM_PATH,
        'finbert_lstm': FINBERT_LSTM_PATH,
        'attention_bilstm': ATTN_BILSTM_PATH
    }

    for name, path in model_files.items():
        if os.path.exists(path):
            try:
                state["models"][name] = keras.models.load_model(path, custom_objects=custom_objs)
                print(f"[OK] Model '{name}' loaded from {path}")
            except Exception as e:
                print(f"[WARN] Error loading {name}: {e}")


@app.on_event("startup")
def startup_event():
    load_assets()


class PredictionRequest(BaseModel):
    prices: List[float] = Field(..., description="Last 10 closing prices (oldest to newest)")
    sentiments: Optional[List[float]] = Field(
        default=None,
        description="Last 10 FinBERT sentiment scores in range [-1, 1]. Defaults to 0.0 if not provided."
    )
    model: Optional[str] = Field(
        default="ensemble",
        description="Choice of 'mlp', 'lstm', 'finbert_lstm', 'attention_bilstm', or 'ensemble'"
    )


class PredictionResponse(BaseModel):
    current_price: float
    predicted_price: float
    price_change: float
    predicted_return_pct: float
    signal: str
    model_used: str


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "scaler_loaded": state["scaler"] is not None,
        "loaded_models": list(state["models"].keys())
    }


@app.get("/model/info")
def model_info():
    available = list(state["models"].keys())
    if len(available) >= 2:
        available.append("ensemble")
    return {
        "sequence_length": state["config"].get("sequence_length", 10),
        "available_models": available,
        "team_contributions": {
            "mlp": "Priyanshu Singh (Model 1: MLP Baseline)",
            "lstm": "Shivli Dimri (Model 2: Standard LSTM)",
            "finbert_lstm": "Dakshayani Sharma (Model 3: FinBERT-LSTM Fusion)",
            "attention_bilstm": "Prajika Saini (Model 4: Attention-BiLSTM)",
            "ensemble": "Joint Team Ensemble (Weighted Blend)"
        }
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    seq_len = state["config"].get("sequence_length", 10)
    if len(request.prices) != seq_len:
        raise HTTPException(status_code=400, detail=f"Expected exactly {seq_len} prices, got {len(request.prices)}")

    sentiments = request.sentiments or [0.0] * seq_len
    if len(sentiments) != seq_len:
        raise HTTPException(status_code=400, detail=f"Expected exactly {seq_len} sentiments, got {len(sentiments)}")

    scaler = state["scaler"]
    if scaler is None:
        raise HTTPException(status_code=500, detail="Price scaler not loaded.")

    prices_arr = np.array(request.prices, dtype=np.float32).reshape(-1, 1)
    scaled_prices = scaler.transform(prices_arr)
    sent_arr = np.array(sentiments, dtype=np.float32).reshape(-1, 1)

    feature_2d = np.concatenate([scaled_prices, sent_arr], axis=1)
    input_2feat = np.expand_dims(feature_2d, axis=0)
    input_1feat = np.expand_dims(scaled_prices, axis=0)

    model_choice = request.model.lower()

    if model_choice == "lstm" and "lstm" in state["models"]:
        pred_scaled = float(state["models"]["lstm"].predict(input_1feat, verbose=0)[0][0])
    elif model_choice == "mlp" and "mlp" in state["models"]:
        pred_scaled = float(state["models"]["mlp"].predict(input_2feat, verbose=0)[0][0])
    elif model_choice == "finbert_lstm" and "finbert_lstm" in state["models"]:
        pred_scaled = float(state["models"]["finbert_lstm"].predict(input_2feat, verbose=0)[0][0])
    elif model_choice == "attention_bilstm" and "attention_bilstm" in state["models"]:
        pred_scaled = float(state["models"]["attention_bilstm"].predict(input_2feat, verbose=0)[0][0])
    else:
        # Ensemble blend
        preds = []
        if "attention_bilstm" in state["models"]:
            preds.append(0.45 * float(state["models"]["attention_bilstm"].predict(input_2feat, verbose=0)[0][0]))
        if "finbert_lstm" in state["models"]:
            preds.append(0.35 * float(state["models"]["finbert_lstm"].predict(input_2feat, verbose=0)[0][0]))
        if "mlp" in state["models"]:
            preds.append(0.20 * float(state["models"]["mlp"].predict(input_2feat, verbose=0)[0][0]))

        if not preds:
            raise HTTPException(status_code=500, detail="No models loaded for prediction.")
        pred_scaled = sum(preds)
        model_choice = "ensemble"

    pred_price = float(scaler.inverse_transform([[pred_scaled]])[0][0])
    current_price = float(request.prices[-1])
    price_change = pred_price - current_price
    pct_change = (price_change / current_price) * 100

    threshold = state["config"].get("threshold_pct", 0.5)
    signal = "BUY" if pct_change > threshold else ("SELL" if pct_change < -threshold else "HOLD")

    return PredictionResponse(
        current_price=round(current_price, 2),
        predicted_price=round(pred_price, 2),
        price_change=round(price_change, 2),
        predicted_return_pct=round(pct_change, 3),
        signal=signal,
        model_used=model_choice
    )


@app.get("/api/backtest/data")
def get_backtest_data(model: str = "ensemble", scope: str = "test"):
    cache_key = f"{model}_{scope}"
    if cache_key in state["cache"]:
        return state["cache"][cache_key]

    if not os.path.exists(STOCK_CSV_PATH) or not os.path.exists(SENTIMENT_CSV_PATH):
        raise HTTPException(status_code=404, detail="Dataset files not found for backtesting.")

    stock_df = pd.read_csv(STOCK_CSV_PATH).iloc[2:].dropna(subset=['Close']).reset_index(drop=True)
    stock_df['Date'] = pd.to_datetime(stock_df['Price' if 'Price' in stock_df.columns else 'Date'], errors='coerce')
    stock_df['Close'] = pd.to_numeric(stock_df['Close'], errors='coerce')
    stock_df = stock_df.dropna(subset=['Date', 'Close'])

    sent_df = pd.read_csv(SENTIMENT_CSV_PATH)
    sent_df['Date'] = pd.to_datetime(sent_df['Date'], errors='coerce')
    sent_col = 'FinBERT score' if 'FinBERT score' in sent_df.columns else ('Sentiment' if 'Sentiment' in sent_df.columns else sent_df.columns[-1])
    sent_df = sent_df.rename(columns={sent_col: 'Sentiment'})
    sent_df = sent_df[['Date', 'Sentiment']].dropna()

    merged = pd.merge(stock_df[['Date', 'Close']], sent_df, on='Date', how='inner').sort_values('Date').reset_index(drop=True)

    dates = merged['Date'].dt.strftime('%Y-%m-%d').values
    stock_close = merged['Close'].values.astype(float)
    sentiment_scores = merged['Sentiment'].values.astype(float)

    scaler = state["scaler"]
    total_len = len(stock_close)
    train_size = int(total_len * 0.6)
    val_size = int(total_len * 0.2)

    if scope == "full":
        sub_stock = stock_close
        sub_sent = sentiment_scores.reshape(-1, 1)
        sub_dates = dates
    else:
        sub_stock = stock_close[train_size + val_size:]
        sub_sent = sentiment_scores[train_size + val_size:].reshape(-1, 1)
        sub_dates = dates[train_size + val_size:]

    sub_stock_scaled = scaler.transform(sub_stock.reshape(-1, 1)).flatten()

    seq_len = 10
    X_2f, X_1f, y_actual, out_dates, out_sent = [], [], [], [], []
    for i in range(len(sub_stock_scaled) - seq_len):
        p_seq = sub_stock_scaled[i:i + seq_len].reshape(-1, 1)
        s_seq = sub_sent[i:i + seq_len]
        X_2f.append(np.concatenate([p_seq, s_seq], axis=1))
        X_1f.append(p_seq)
        y_actual.append(float(sub_stock[i + seq_len]))
        out_dates.append(str(sub_dates[i + seq_len]))
        out_sent.append(float(sub_sent[i + seq_len][0]))

    X_2f = np.array(X_2f)
    X_1f = np.array(X_1f)

    # Predictions
    m_choice = model.lower()
    if m_choice == "lstm" and "lstm" in state["models"]:
        preds_scaled = state["models"]["lstm"].predict(X_1f, verbose=0).flatten()
    elif m_choice == "mlp" and "mlp" in state["models"]:
        preds_scaled = state["models"]["mlp"].predict(X_2f, verbose=0).flatten()
    elif m_choice == "finbert_lstm" and "finbert_lstm" in state["models"]:
        preds_scaled = state["models"]["finbert_lstm"].predict(X_2f, verbose=0).flatten()
    elif m_choice == "attention_bilstm" and "attention_bilstm" in state["models"]:
        preds_scaled = state["models"]["attention_bilstm"].predict(X_2f, verbose=0).flatten()
    else:
        # Ensemble
        preds_scaled = np.zeros(len(X_2f))
        weights = 0.0
        if "attention_bilstm" in state["models"]:
            preds_scaled += 0.45 * state["models"]["attention_bilstm"].predict(X_2f, verbose=0).flatten()
            weights += 0.45
        if "finbert_lstm" in state["models"]:
            preds_scaled += 0.35 * state["models"]["finbert_lstm"].predict(X_2f, verbose=0).flatten()
            weights += 0.35
        if "mlp" in state["models"]:
            preds_scaled += 0.20 * state["models"]["mlp"].predict(X_2f, verbose=0).flatten()
            weights += 0.20
        if weights > 0:
            preds_scaled /= weights

    preds_unscaled = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten().tolist()

    # Backtesting simulation
    capital = 10000.0
    shares = 0.0
    portfolio_history = []
    signals = []
    trades = []
    wins = 0
    trade_returns = []
    last_buy_price = None

    for i in range(len(preds_unscaled)):
        cur_p = y_actual[i]
        pred_p = preds_unscaled[i]
        date_str = out_dates[i]

        pred_ret = (pred_p - cur_p) / cur_p
        sig = "HOLD"

        if pred_ret > 0.005 and capital > 0:
            buy_shares = capital / cur_p
            shares += buy_shares
            last_buy_price = cur_p
            trades.append({"date": date_str, "type": "BUY", "price": round(cur_p, 2), "shares": round(buy_shares, 2)})
            capital = 0.0
            sig = "BUY"
        elif pred_ret < -0.005 and shares > 0:
            sell_amount = shares * cur_p
            if last_buy_price is not None:
                ret_val = (cur_p - last_buy_price) / last_buy_price
                trade_returns.append(ret_val)
                if ret_val > 0:
                    wins += 1
            trades.append({"date": date_str, "type": "SELL", "price": round(cur_p, 2), "shares": round(shares, 2)})
            capital = sell_amount
            shares = 0.0
            sig = "SELL"

        current_val = capital + (shares * cur_p)
        portfolio_history.append(round(current_val, 2))
        signals.append(sig)

    final_val = portfolio_history[-1] if portfolio_history else 10000.0
    tot_ret = ((final_val - 10000.0) / 10000.0) * 100
    bh_ret = ((y_actual[-1] - y_actual[0]) / y_actual[0]) * 100
    win_rate = (wins / len(trade_returns) * 100) if len(trade_returns) > 0 else 0.0

    daily_pct = np.diff(portfolio_history) / portfolio_history[:-1] if len(portfolio_history) > 1 else np.array([0.0])
    sharpe = float((np.mean(daily_pct) / np.std(daily_pct)) * np.sqrt(252)) if np.std(daily_pct) > 0 else 0.0

    result = {
        "dates": out_dates,
        "actual_prices": [round(p, 2) for p in y_actual],
        "predicted_prices": [round(p, 2) for p in preds_unscaled],
        "portfolio_history": portfolio_history,
        "sentiment_scores": [round(s, 3) for s in out_sent],
        "signals": signals,
        "trades": trades,
        "metrics": {
            "initial_capital": 10000.0,
            "final_equity": round(final_val, 2),
            "strategy_return_pct": round(tot_ret, 2),
            "buy_hold_return_pct": round(bh_ret, 2),
            "sharpe_ratio": round(sharpe, 2),
            "win_rate_pct": round(win_rate, 2),
            "total_trades": len(trades),
            "model_name": model.upper(),
            "scope_name": f"{len(out_dates)} Trading Days ({'Test Set' if scope=='test' else 'Full History'})"
        }
    }
    state["cache"][cache_key] = result
    return result


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FinBERT-LSTM Multi-Modal Trading Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        :root {
            --bg-main: #0b0f19;
            --card-bg: rgba(22, 30, 49, 0.85);
            --border-col: rgba(255, 255, 255, 0.08);
            --accent: #6366f1;
            --green: #10b981;
            --red: #ef4444;
            --yellow: #f59e0b;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', system-ui, sans-serif; }
        body { background: var(--bg-main); color: var(--text-main); min-height: 100vh; padding: 24px; }
        .container { max-width: 1440px; margin: 0 auto; }
        header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; border-bottom: 1px solid var(--border-col); padding-bottom: 16px; }
        .logo { font-size: 22px; font-weight: 700; display: flex; align-items: center; gap: 10px; color: #fff; }
        .badge-live { background: rgba(16, 185, 129, 0.15); color: var(--green); border: 1px solid var(--green); padding: 4px 12px; border-radius: 999px; font-size: 13px; font-weight: 600; display: flex; align-items: center; gap: 6px; }
        .badge-live::before { content: ''; width: 8px; height: 8px; background: var(--green); border-radius: 50%; display: inline-block; animation: pulse 1.5s infinite; }
        @keyframes pulse { 0% { opacity: 0.4; } 50% { opacity: 1; } 100% { opacity: 0.4; } }
        
        .controls-panel { background: var(--card-bg); border: 1px solid var(--border-col); border-radius: 14px; padding: 18px 24px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 16px; margin-bottom: 24px; }
        .control-group { display: flex; align-items: center; gap: 12px; }
        button.btn-primary { background: linear-gradient(135deg, #6366f1, #4f46e5); color: white; border: none; padding: 10px 22px; border-radius: 8px; font-weight: 600; cursor: pointer; display: flex; align-items: center; gap: 8px; transition: transform 0.15s ease; }
        button.btn-primary:hover { transform: translateY(-1px); filter: brightness(1.1); }
        button.btn-secondary { background: rgba(255,255,255,0.08); color: white; border: 1px solid var(--border-col); padding: 10px 18px; border-radius: 8px; font-weight: 600; cursor: pointer; display: flex; align-items: center; gap: 8px; }
        button.btn-secondary:hover { background: rgba(255,255,255,0.12); }
        select, input { background: rgba(11, 15, 25, 0.8); border: 1px solid var(--border-col); color: white; padding: 8px 14px; border-radius: 8px; outline: none; }
        
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 24px; }
        .stat-card { background: var(--card-bg); border: 1px solid var(--border-col); border-radius: 12px; padding: 16px 20px; }
        .stat-label { font-size: 13px; color: var(--text-muted); margin-bottom: 6px; display: flex; align-items: center; gap: 6px; }
        .stat-val { font-size: 24px; font-weight: 700; color: #fff; }
        .stat-sub { font-size: 12px; margin-top: 4px; color: var(--text-muted); }
        .text-green { color: var(--green); }
        .text-red { color: var(--red); }
        .badge-signal { display: inline-block; padding: 4px 12px; border-radius: 6px; font-size: 14px; font-weight: 700; }
        .badge-buy { background: rgba(16, 185, 129, 0.2); color: var(--green); border: 1px solid var(--green); }
        .badge-sell { background: rgba(239, 68, 68, 0.2); color: var(--red); border: 1px solid var(--red); }
        .badge-hold { background: rgba(245, 158, 11, 0.2); color: var(--yellow); border: 1px solid var(--yellow); }

        .charts-grid { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; margin-bottom: 24px; }
        @media(max-width: 1024px) { .charts-grid { grid-template-columns: 1fr; } }
        .chart-box { background: var(--card-bg); border: 1px solid var(--border-col); border-radius: 14px; padding: 20px; }
        .chart-title { font-size: 16px; font-weight: 600; margin-bottom: 14px; display: flex; justify-content: space-between; align-items: center; }

        .trade-log { max-height: 280px; overflow-y: auto; font-size: 13px; }
        .trade-item { display: flex; justify-content: space-between; padding: 8px 12px; border-bottom: 1px solid rgba(255,255,255,0.04); align-items: center; }
        .progress-bar-container { width: 100%; height: 6px; background: rgba(255,255,255,0.06); border-radius: 999px; margin-top: 14px; overflow: hidden; }
        .progress-bar { height: 100%; width: 0%; background: linear-gradient(90deg, #6366f1, #10b981); transition: width 0.1s linear; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo">
                <i class="fa-solid fa-chart-line" style="color: #6366f1;"></i>
                FinBERT-LSTM Quantitative Trading System (Team 12)
            </div>
            <div style="display: flex; gap: 12px; align-items: center;">
                <a href="/docs" target="_blank" style="color: var(--text-muted); font-size: 14px; text-decoration: none;">
                    <i class="fa-solid fa-book"></i> API Documentation
                </a>
                <div class="badge-live">System Active</div>
            </div>
        </header>

        <div class="controls-panel">
            <div class="control-group">
                <label style="font-size: 14px; color: var(--text-muted); font-weight: 600;">Model Architecture:</label>
                <select id="modelSelect">
                    <option value="ensemble" selected>Quantitative Ensemble (Blended)</option>
                    <option value="attention_bilstm">Model 4: Attention-BiLSTM (Prajika)</option>
                    <option value="finbert_lstm">Model 3: FinBERT-LSTM Fusion (Dakshayani)</option>
                    <option value="lstm">Model 2: Standard LSTM (Shivli)</option>
                    <option value="mlp">Model 1: MLP Baseline (Priyanshu)</option>
                </select>
                <select id="scopeSelect">
                    <option value="test" selected>Out-of-Sample Test Set (86 Days)</option>
                    <option value="full">Full Aligned History (468 Days)</option>
                </select>
            </div>
            <div class="control-group">
                <button class="btn-primary" id="btnPlay" onclick="startSimulation()">
                    <i class="fa-solid fa-play"></i> Start Live Backtest
                </button>
                <button class="btn-secondary" id="btnReset" onclick="resetSimulation()">
                    <i class="fa-solid fa-rotate-right"></i> Reset
                </button>
            </div>
        </div>

        <div class="progress-bar-container">
            <div class="progress-bar" id="progressBar"></div>
        </div>

        <div class="stats-grid" style="margin-top: 18px;">
            <div class="stat-card">
                <div class="stat-label"><i class="fa-solid fa-wallet"></i> Portfolio Equity</div>
                <div class="stat-val" id="valPortfolio">$10,000.00</div>
                <div class="stat-sub" id="valReturn">+0.00% Net Return</div>
            </div>
            <div class="stat-card">
                <div class="stat-label"><i class="fa-solid fa-tags"></i> Current NASDAQ Price</div>
                <div class="stat-val" id="valPrice">---</div>
                <div class="stat-sub" id="valDate">Simulation Ready</div>
            </div>
            <div class="stat-card">
                <div class="stat-label"><i class="fa-solid fa-bullseye"></i> AI Model Prediction</div>
                <div class="stat-val" id="valPred">---</div>
                <div class="stat-sub" id="valSignal"><span class="badge-signal badge-hold">HOLD</span></div>
            </div>
            <div class="stat-card">
                <div class="stat-label"><i class="fa-solid fa-scale-balanced"></i> Benchmark (Buy & Hold)</div>
                <div class="stat-val" id="valBHReturn">---</div>
                <div class="stat-sub" id="valSharpe">Sharpe: 0.00 | Win: 0.0%</div>
            </div>
        </div>

        <div class="charts-grid">
            <div class="chart-box">
                <div class="chart-title">
                    <span><i class="fa-solid fa-chart-candlestick"></i> Price Forecasting & AI Trajectory</span>
                    <span id="labelIndex" style="font-size: 13px; color: var(--text-muted);">Step 0 / 0</span>
                </div>
                <canvas id="priceChart" height="280"></canvas>
            </div>
            <div class="chart-box">
                <div class="chart-title">
                    <span><i class="fa-solid fa-money-bill-trend-up"></i> Cumulative Strategy Equity ($)</span>
                </div>
                <canvas id="equityChart" height="280"></canvas>
            </div>
        </div>

        <div class="charts-grid" style="grid-template-columns: 1fr 1fr;">
            <div class="chart-box">
                <div class="chart-title">
                    <span><i class="fa-solid fa-newspaper"></i> FinBERT Daily News Sentiment Score</span>
                </div>
                <canvas id="sentimentChart" height="180"></canvas>
            </div>
            <div class="chart-box">
                <div class="chart-title">
                    <span><i class="fa-solid fa-list-check"></i> Executed Algorithmic Trades</span>
                    <span id="tradeCount" style="font-size: 13px; color: var(--text-muted);">0 Trades</span>
                </div>
                <div class="trade-log" id="tradeLog">
                    <div style="text-align: center; color: var(--text-muted); padding: 40px;">Click "Start Live Backtest" to begin trading simulation.</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let rawData = null;
        let simInterval = null;
        let currentIdx = 0;
        let priceChart, equityChart, sentimentChart;

        function initCharts() {
            const chartOpts = {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                scales: {
                    x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#9ca3af', maxTicksLimit: 8 } },
                    y: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#9ca3af' } }
                },
                plugins: { legend: { labels: { color: '#f3f4f6' } } }
            };

            const ctxP = document.getElementById('priceChart').getContext('2d');
            priceChart = new Chart(ctxP, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        { label: 'Actual NASDAQ-100 Close', data: [], borderColor: '#f3f4f6', borderWidth: 2, pointRadius: 0 },
                        { label: 'AI Predicted Close', data: [], borderColor: '#6366f1', borderWidth: 2, borderDash: [4, 4], pointRadius: 0 }
                    ]
                },
                options: chartOpts
            });

            const ctxE = document.getElementById('equityChart').getContext('2d');
            equityChart = new Chart(ctxE, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        { label: 'AI Strategy ($)', data: [], borderColor: '#10b981', borderWidth: 2, fill: true, backgroundColor: 'rgba(16, 185, 129, 0.08)', pointRadius: 0 }
                    ]
                },
                options: chartOpts
            });

            const ctxS = document.getElementById('sentimentChart').getContext('2d');
            sentimentChart = new Chart(ctxS, {
                type: 'bar',
                data: {
                    labels: [],
                    datasets: [
                        { label: 'FinBERT Score', data: [], backgroundColor: [] }
                    ]
                },
                options: {
                    ...chartOpts,
                    scales: {
                        ...chartOpts.scales,
                        y: { min: -1.0, max: 1.0, ticks: { color: '#9ca3af' } }
                    }
                }
            });
        }

        async function loadBacktestData() {
            const m = document.getElementById('modelSelect').value;
            const s = document.getElementById('scopeSelect').value;
            const res = await fetch(`/api/backtest/data?model=${m}&scope=${s}`);
            rawData = await res.json();
            resetSimulation();
        }

        function resetSimulation() {
            clearInterval(simInterval);
            simInterval = null;
            currentIdx = 0;
            document.getElementById('btnPlay').innerHTML = '<i class="fa-solid fa-play"></i> Start Live Backtest';
            document.getElementById('progressBar').style.width = '0%';
            document.getElementById('tradeLog').innerHTML = '';
            document.getElementById('valPortfolio').innerText = '$10,000.00';
            document.getElementById('valReturn').innerText = '+0.00% Net Return';
            document.getElementById('valReturn').className = 'stat-sub';
            document.getElementById('valPrice').innerText = '---';
            document.getElementById('valPred').innerText = '---';
            document.getElementById('valDate').innerText = 'Simulation Ready';
            document.getElementById('valBHReturn').innerText = '---';
            document.getElementById('tradeCount').innerText = '0 Trades';

            if (!rawData) return;
            priceChart.data.labels = [];
            priceChart.data.datasets[0].data = [];
            priceChart.data.datasets[1].data = [];
            priceChart.update();

            equityChart.data.labels = [];
            equityChart.data.datasets[0].data = [];
            equityChart.update();

            sentimentChart.data.labels = [];
            sentimentChart.data.datasets[0].data = [];
            sentimentChart.data.datasets[0].backgroundColor = [];
            sentimentChart.update();
        }

        function startSimulation() {
            if (!rawData) return;
            if (simInterval) {
                clearInterval(simInterval);
                simInterval = null;
                document.getElementById('btnPlay').innerHTML = '<i class="fa-solid fa-play"></i> Resume';
                return;
            }
            document.getElementById('btnPlay').innerHTML = '<i class="fa-solid fa-pause"></i> Pause';
            simInterval = setInterval(stepSimulation, 80);
        }

        function stepSimulation() {
            if (!rawData || currentIdx >= rawData.dates.length) {
                clearInterval(simInterval);
                simInterval = null;
                document.getElementById('btnPlay').innerHTML = '<i class="fa-solid fa-check"></i> Finished';
                return;
            }

            const i = currentIdx;
            const date = rawData.dates[i];
            const actualP = rawData.actual_prices[i];
            const predP = rawData.predicted_prices[i];
            const equity = rawData.portfolio_history[i];
            const sent = rawData.sentiment_scores[i];
            const sig = rawData.signals[i];

            // Update charts
            priceChart.data.labels.push(date);
            priceChart.data.datasets[0].data.push(actualP);
            priceChart.data.datasets[1].data.push(predP);
            priceChart.update();

            equityChart.data.labels.push(date);
            equityChart.data.datasets[0].data.push(equity);
            equityChart.update();

            sentimentChart.data.labels.push(date);
            sentimentChart.data.datasets[0].data.push(sent);
            sentimentChart.data.datasets[0].backgroundColor.push(sent >= 0 ? 'rgba(16,185,129,0.7)' : 'rgba(239,68,68,0.7)');
            sentimentChart.update();

            // Update stats
            document.getElementById('valPortfolio').innerText = `$${equity.toLocaleString('en-US', {minimumFractionDigits:2, maximumFractionDigits:2})}`;
            const ret = ((equity - 10000) / 10000) * 100;
            document.getElementById('valReturn').innerText = `${ret >= 0 ? '+' : ''}${ret.toFixed(2)}% Net Return`;
            document.getElementById('valReturn').className = `stat-sub ${ret >= 0 ? 'text-green' : 'text-red'}`;
            document.getElementById('valPrice').innerText = `$${actualP.toFixed(2)}`;
            document.getElementById('valPred').innerText = `$${predP.toFixed(2)}`;
            document.getElementById('valDate').innerText = date;

            let sigClass = 'badge-hold';
            if (sig === 'BUY') sigClass = 'badge-buy';
            if (sig === 'SELL') sigClass = 'badge-sell';
            document.getElementById('valSignal').innerHTML = `<span class="badge-signal ${sigClass}">${sig}</span>`;

            // Progress bar
            const pct = ((i + 1) / rawData.dates.length) * 100;
            document.getElementById('progressBar').style.width = `${pct}%`;
            document.getElementById('labelIndex').innerText = `Day ${i + 1} / ${rawData.dates.length}`;

            // Check trade log
            const trade = rawData.trades.find(t => t.date === date);
            if (trade) {
                const log = document.getElementById('tradeLog');
                const div = document.createElement('div');
                div.className = 'trade-item';
                div.innerHTML = `
                    <div><strong>${trade.date}</strong> - ${trade.shares} units</div>
                    <div><span class="badge-signal ${trade.type === 'BUY' ? 'badge-buy' : 'badge-sell'}">${trade.type}</span> @ $${trade.price}</div>
                `;
                log.prepend(div);
                document.getElementById('tradeCount').innerText = `${document.querySelectorAll('.trade-item').length} Trades`;
            }

            // Final metrics on last step
            if (i === rawData.dates.length - 1) {
                const m = rawData.metrics;
                document.getElementById('valBHReturn').innerText = `${m.buy_hold_return_pct >= 0 ? '+' : ''}${m.buy_hold_return_pct}% B&H`;
                document.getElementById('valSharpe').innerText = `Sharpe: ${m.sharpe_ratio} | Win Rate: ${m.win_rate_pct}%`;
            }

            currentIdx++;
        }

        window.onload = () => {
            initCharts();
            loadBacktestData();
            document.getElementById('modelSelect').addEventListener('change', loadBacktestData);
            document.getElementById('scopeSelect').addEventListener('change', loadBacktestData);
        };
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return HTMLResponse(content=DASHBOARD_HTML)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
