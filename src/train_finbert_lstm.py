"""
FinBERT-LSTM: Sentiment-Augmented Sequential Deep Learning Model
==================================================================
Contribution: Dakshayani Sharma (230911066)

Architecture: Stacked LSTM backbone (32 -> 16 units) augmented with
FinBERT daily financial news sentiment scores.

Key features:
  - Multimodal input at every timestep: [price_t, sentiment_t] pairs
    merged on Date from stock price and FinBERT sentiment datasets.
  - Apples-to-apples ablation: trains both multimodal FinBERT-LSTM and
    matching price-only ablation on the exact same date range & splits.
  - Strict chronological 60/20/20 train/val/test split with train-only scaling.
  - Full metrics export to results/finbert_lstm_metrics.json and results/.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

# Set local cache paths to avoid sandbox permission errors
os.environ.setdefault("KERAS_HOME", os.path.join(PROJECT_ROOT, ".keras"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(PROJECT_ROOT, ".matplotlib"))
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)

STOCK_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'stock_price.csv')
SENTIMENT_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'sentiment_new.csv')
if not os.path.exists(SENTIMENT_PATH):
    SENTIMENT_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'sentiment.csv')

MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

SEQUENCE_LENGTH = 10


def load_merged_data():
    """Load price + FinBERT sentiment, merged on Date (inner join)."""
    stock_df = pd.read_csv(STOCK_PATH)
    stock_df = stock_df.iloc[2:].copy().reset_index(drop=True)
    stock_df = stock_df.rename(columns={'Price': 'Date'})
    for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
        stock_df[col] = pd.to_numeric(stock_df[col], errors='coerce')
    stock_df['Date'] = pd.to_datetime(stock_df['Date'], errors='coerce')
    stock_df = stock_df.dropna(subset=['Date', 'Close'])

    sent_df = pd.read_csv(SENTIMENT_PATH)
    sent_df['Date'] = pd.to_datetime(sent_df['Date'], errors='coerce')
    sent_col = 'FinBERT score' if 'FinBERT score' in sent_df.columns else ('Sentiment' if 'Sentiment' in sent_df.columns else sent_df.columns[-1])
    sent_df = sent_df.rename(columns={sent_col: 'Sentiment'})
    sent_df = sent_df[['Date', 'Sentiment']].dropna(subset=['Date', 'Sentiment'])

    merged = pd.merge(stock_df[['Date', 'Close']], sent_df, on='Date', how='inner')
    merged = merged.sort_values('Date').reset_index(drop=True)
    return merged


def create_sequences(data, seq_len=SEQUENCE_LENGTH):
    """data: 2D array (n_timesteps, n_features). Target is next-step Close (feature 0)."""
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len, 0])
    return np.array(X), np.array(y)


def build_model(n_features):
    model = keras.Sequential([
        layers.Input(shape=(SEQUENCE_LENGTH, n_features)),
        layers.LSTM(32, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), name="lstm_1"),
        layers.Dropout(0.3, name="dropout_1"),
        layers.LSTM(16, kernel_regularizer=regularizers.l2(0.001), name="lstm_2"),
        layers.Dropout(0.2, name="dropout_2"),
        layers.Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.001), name="dense_1"),
        layers.Dense(1, activation='linear', name="price_prediction")
    ], name=f"FinBERT_LSTM_{n_features}feat")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model


def run_experiment(merged, use_sentiment: bool, tag: str):
    print("\n" + "=" * 80)
    print(f"RUN: {tag}  (use_sentiment={use_sentiment})")
    print("=" * 80)

    feature_cols = ['Close', 'Sentiment'] if use_sentiment else ['Close']
    data = merged[feature_cols].values.astype(float)
    n_features = data.shape[1]

    total_len = len(data)
    train_size = int(total_len * 0.6)
    val_size = int(total_len * 0.2)

    train_raw = data[:train_size]
    val_raw = data[train_size:train_size + val_size]
    test_raw = data[train_size + val_size:]

    print(f"  Samples: {total_len}  Train: {len(train_raw)}  Val: {len(val_raw)}  Test: {len(test_raw)}")

    # Fit scalers on train only, per feature, to prevent leakage.
    scalers = [MinMaxScaler() for _ in range(n_features)]
    train_scaled = np.zeros_like(train_raw)
    val_scaled = np.zeros_like(val_raw)
    test_scaled = np.zeros_like(test_raw)
    for j in range(n_features):
        train_scaled[:, j] = scalers[j].fit_transform(train_raw[:, j].reshape(-1, 1)).flatten()
        val_scaled[:, j] = scalers[j].transform(val_raw[:, j].reshape(-1, 1)).flatten()
        test_scaled[:, j] = scalers[j].transform(test_raw[:, j].reshape(-1, 1)).flatten()
    price_scaler = scalers[0]
    if tag == 'finbert_lstm':
        import pickle
        scaler_path = os.path.join(MODELS_DIR, 'price_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(price_scaler, f)
        print(f"  [OK] Saved price scaler to {scaler_path}")

    X_train, y_train = create_sequences(train_scaled)
    X_val, y_val = create_sequences(val_scaled)
    X_test, y_test = create_sequences(test_scaled)
    print(f"  Sequences: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

    model = build_model(n_features)
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5, verbose=1)
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=120,
        batch_size=16,
        callbacks=callbacks,
        verbose=1
    )

    model_path = os.path.join(MODELS_DIR, f'{tag}_model.keras')
    model.save(model_path)
    print(f"  [OK] Saved model to {model_path}")

    y_pred_scaled = model.predict(X_test, verbose=0).flatten()
    y_test_unscaled = price_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_unscaled = price_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    mae = mean_absolute_error(y_test_unscaled, y_pred_unscaled)
    rmse = float(np.sqrt(np.mean((y_test_unscaled - y_pred_unscaled) ** 2)))
    mape = mean_absolute_percentage_error(y_test_unscaled, y_pred_unscaled) * 100
    r2 = r2_score(y_test_unscaled, y_pred_unscaled)

    actual_direction = np.diff(y_test_unscaled) > 0
    pred_direction = np.diff(y_pred_unscaled) > 0
    dir_accuracy = np.mean(actual_direction == pred_direction) * 100

    signals = np.zeros(len(y_pred_unscaled) - 1)
    for i in range(len(signals)):
        signals[i] = 1 if y_pred_unscaled[i + 1] > y_test_unscaled[i] else 0

    actual_pct_change = np.diff(y_test_unscaled) / y_test_unscaled[:-1]
    strategy_returns = signals * actual_pct_change
    cumulative_strategy_return = float(np.prod(1 + strategy_returns) - 1)
    cumulative_buy_hold = float(np.prod(1 + actual_pct_change) - 1)
    sharpe_ratio = float((np.mean(strategy_returns) / np.std(strategy_returns)) * np.sqrt(252)) if np.std(strategy_returns) > 0 else 0.0
    win_rate = float((strategy_returns > 0).sum() / len(strategy_returns) * 100) if len(strategy_returns) > 0 else 0.0

    display_name = 'FinBERT-LSTM Fusion' if use_sentiment else 'FinBERT-LSTM (Price-Only Ablation)'
    metrics = {
        'model_name': display_name,
        'tag': tag,
        'member_owner': 'Dakshayani Sharma (230911066)',
        'architecture': f'Stacked LSTM 32 -> 16 ({n_features} features, L2=0.001, Dropout)',
        'input_features': feature_cols,
        'use_sentiment': use_sentiment,
        'n_features': n_features,
        'samples_total': int(total_len),
        'samples_train': int(len(train_raw)),
        'samples_val': int(len(val_raw)),
        'samples_test': int(len(test_raw)),
        'sequence_length': SEQUENCE_LENGTH,
        'mae': float(mae),
        'rmse': float(rmse),
        'mape_pct': float(mape),
        'r2': float(r2),
        'directional_accuracy_pct': float(dir_accuracy),
        'strategy_return_pct': float(cumulative_strategy_return * 100),
        'buy_hold_return_pct': float(cumulative_buy_hold * 100),
        'sharpe_ratio': float(sharpe_ratio),
        'win_rate_pct': float(win_rate),
        'model_path': model_path,
    }

    print("\n" + "=" * 50)
    print(f"{tag.upper()} PERFORMANCE REPORT")
    print("=" * 50)
    for k in ['mae', 'rmse', 'mape_pct', 'r2', 'directional_accuracy_pct',
              'strategy_return_pct', 'buy_hold_return_pct', 'sharpe_ratio', 'win_rate_pct']:
        print(f"  {k:28s}: {metrics[k]:.4f}")
    print("=" * 50)

    metrics_path = os.path.join(RESULTS_DIR, f'{tag}_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  [OK] Saved metrics to {metrics_path}")

    plot_path = os.path.join(RESULTS_DIR, f'{tag}_results.png')
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(y_test_unscaled, label='Actual NDX Close', color='black', linewidth=1.8)
    plt.plot(y_pred_unscaled, label=f'{tag} Predictions', color='#dc2626', linestyle='--', linewidth=1.5)
    plt.title(f'{display_name} vs Actual (MAE: ${mae:.1f})', fontsize=12, fontweight='bold')
    plt.xlabel('Timestep (Test Set Days)')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    strat_cum = np.cumprod(1 + strategy_returns)
    bh_cum = np.cumprod(1 + actual_pct_change)
    plt.plot(strat_cum, label=f'{tag} Strategy (+{cumulative_strategy_return*100:.1f}%)', color='#10b981', linewidth=1.8)
    plt.plot(bh_cum, label=f'Buy & Hold (+{cumulative_buy_hold*100:.1f}%)', color='#64748b', linestyle=':', linewidth=1.5)
    plt.title(f'Backtest Equity Curve (Sharpe: {sharpe_ratio:.2f})', fontsize=12, fontweight='bold')
    plt.xlabel('Trading Days')
    plt.ylabel('Cumulative Multiple')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  [OK] Saved results plot to {plot_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="FinBERT-LSTM training (sentiment + price-only ablation)")
    parser.add_argument('--only', choices=['both', 'sentiment', 'priceonly'], default='both',
                        help="Which run(s) to execute. Default trains both for a direct comparison.")
    args = parser.parse_args()

    print("=" * 80)
    print("FinBERT-LSTM - Training & Evaluation Pipeline (Dakshayani)")
    print("=" * 80)
    print(f"\nLoading & merging price ({STOCK_PATH}) with sentiment ({SENTIMENT_PATH})...")
    merged = load_merged_data()
    print(f"  Merged (inner join on Date): {len(merged)} rows "
          f"[{merged['Date'].min().date()} -> {merged['Date'].max().date()}]")

    all_metrics = {}
    if args.only in ('both', 'sentiment'):
        all_metrics['finbert_lstm'] = run_experiment(merged, use_sentiment=True, tag='finbert_lstm')
    if args.only in ('both', 'priceonly'):
        all_metrics['finbert_lstm_priceonly'] = run_experiment(merged, use_sentiment=False, tag='finbert_lstm_priceonly')

    if len(all_metrics) == 2:
        summary_path = os.path.join(RESULTS_DIR, 'finbert_lstm_comparison.json')
        with open(summary_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\n[OK] Saved sentiment-vs-price-only comparison to {summary_path}")
        print("\n" + "=" * 80)
        print("SENTIMENT ABLATION SUMMARY (same merged date range & split)")
        print("=" * 80)
        s, p = all_metrics['finbert_lstm'], all_metrics['finbert_lstm_priceonly']
        print(f"  {'Metric':28s} {'+Sentiment':>14s} {'Price-only':>14s}")
        for k, label in [('mae', 'MAE ($)'), ('rmse', 'RMSE ($)'), ('mape_pct', 'MAPE (%)'),
                         ('r2', 'R2'), ('directional_accuracy_pct', 'Directional Acc (%)'),
                         ('sharpe_ratio', 'Sharpe'), ('strategy_return_pct', 'Return (%)')]:
            print(f"  {label:28s} {s[k]:14.4f} {p[k]:14.4f}")
        print("=" * 80)

    print("\n[DONE] FinBERT-LSTM training and evaluation complete.")


if __name__ == '__main__':
    main()
