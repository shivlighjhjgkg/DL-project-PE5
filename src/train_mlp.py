"""
MLP (Multi-Layer Perceptron) Baseline Trading Model
===================================================
Contribution: Baseline Deep Neural Network Architecture
Key Features:
1. Multi-layer Feedforward Architecture (Dense 64 -> 32 -> 16 -> 1)
2. Dropout and L2 Weight Regularization to prevent overfitting on small datasets
3. Sequence windowing with Price and FinBERT sentiment features
4. Full Backtest simulation: MAE, MAPE, R2, Directional Accuracy, Sharpe Ratio
5. Model checkpoint export to models/mlp_model.keras
"""

import os
import shutil
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)

print("=" * 80)
print("MLP BASELINE TRADING MODEL - Training & Evaluation Pipeline")
print("=" * 80)

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

STOCK_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'stock_price.csv') if os.path.exists(os.path.join(PROJECT_ROOT, 'data', 'raw', 'stock_price.csv')) else 'stock_price.csv'
SENTIMENT_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'sentiment.csv') if os.path.exists(os.path.join(PROJECT_ROOT, 'data', 'processed', 'sentiment.csv')) else 'sentiment.csv'
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, 'models', 'mlp_model.keras') if os.path.exists(os.path.join(PROJECT_ROOT, 'models')) else 'mlp_model.keras'
API_MODEL_PATH = os.path.join(PROJECT_ROOT, 'api', 'mlp_model.keras')
PLOT_SAVE_PATH = os.path.join(PROJECT_ROOT, 'results', 'mlp_model_results.png') if os.path.exists(os.path.join(PROJECT_ROOT, 'results')) else 'mlp_model_results.png'

print(f"\n[1/7] Loading dataset from {STOCK_PATH}...")
stock_df = pd.read_csv(STOCK_PATH)
sentiment_df = pd.read_csv(SENTIMENT_PATH)

stock_df = stock_df.iloc[2:].copy().reset_index(drop=True)
for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
    stock_df[col] = pd.to_numeric(stock_df[col], errors='coerce')
stock_df = stock_df.dropna()

stock_close = stock_df['Close'].values
sentiment_scores = sentiment_df['FinBERT score'].values

print(f"  Samples: {len(stock_close)}")
print(f"  Close Price Range: [{stock_close.min():.2f}, {stock_close.max():.2f}]")
print(f"  Sentiment Range: [{sentiment_scores.min():.3f}, {sentiment_scores.max():.3f}]")

print("\n[2/7] Splitting data (60% Train, 20% Val, 20% Test)...")
total_len = len(stock_close)
train_size = int(total_len * 0.6)
val_size = int(total_len * 0.2)

train_stock = stock_close[:train_size]
val_stock = stock_close[train_size:train_size + val_size]
test_stock = stock_close[train_size + val_size:]

train_sentiment = sentiment_scores[:train_size]
val_sentiment = sentiment_scores[train_size:train_size + val_size]
test_sentiment = sentiment_scores[train_size + val_size:]

print(f"  Train: {len(train_stock)}, Val: {len(val_stock)}, Test: {len(test_stock)}")

print("\n[3/7] Scaling features (Train-only fit to prevent data leakage)...")
price_scaler = MinMaxScaler()
train_stock_scaled = price_scaler.fit_transform(train_stock.reshape(-1, 1)).flatten()
val_stock_scaled = price_scaler.transform(val_stock.reshape(-1, 1)).flatten()
test_stock_scaled = price_scaler.transform(test_stock.reshape(-1, 1)).flatten()

sequence_length = 10

def create_sequences(stock_data, sentiment_data, seq_len=10):
    X, y = [], []
    for i in range(len(stock_data) - seq_len):
        stock_seq = stock_data[i:i + seq_len].reshape(-1, 1)
        sent_seq = sentiment_data[i:i + seq_len].reshape(-1, 1)
        seq = np.concatenate([stock_seq, sent_seq], axis=1)
        X.append(seq)
        y.append(stock_data[i + seq_len])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_stock_scaled, train_sentiment, sequence_length)
X_val, y_val = create_sequences(val_stock_scaled, val_sentiment, sequence_length)
X_test, y_test = create_sequences(test_stock_scaled, test_sentiment, sequence_length)

print(f"  Sequences created: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

print("\n[4/7] Building MLP architecture...")
mlp_model = keras.Sequential([
    layers.Input(shape=(sequence_length, 2)),
    layers.Flatten(name="flatten_sequence"),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001), name="dense_1"),
    layers.Dropout(0.3, name="dropout_1"),
    layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001), name="dense_2"),
    layers.Dropout(0.2, name="dropout_2"),
    layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001), name="dense_3"),
    layers.Dense(1, activation='linear', name="price_prediction")
], name="MLP_Baseline_Model")

mlp_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)
mlp_model.summary()

print("\n[5/7] Training MLP baseline model...")
callbacks = [
    EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5, verbose=1)
]

history = mlp_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=120,
    batch_size=16,
    callbacks=callbacks,
    verbose=1
)

os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
mlp_model.save(MODEL_SAVE_PATH)
print(f"  [OK] Saved model to {MODEL_SAVE_PATH}")

if os.path.exists(os.path.dirname(API_MODEL_PATH)):
    shutil.copy(MODEL_SAVE_PATH, API_MODEL_PATH)
    print(f"  [OK] Synced model to {API_MODEL_PATH}")

print("\n[6/7] Evaluating MLP on unseen test data...")
y_pred_scaled = mlp_model.predict(X_test, verbose=0).flatten()

y_test_unscaled = price_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred_unscaled = price_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

mae = mean_absolute_error(y_test_unscaled, y_pred_unscaled)
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
cumulative_strategy_return = np.prod(1 + strategy_returns) - 1
cumulative_buy_hold = np.prod(1 + actual_pct_change) - 1

if np.std(strategy_returns) > 0:
    sharpe_ratio = (np.mean(strategy_returns) / np.std(strategy_returns)) * np.sqrt(252)
else:
    sharpe_ratio = 0.0

print("\n" + "=" * 50)
print("MLP PERFORMANCE REPORT")
print("=" * 50)
print(f"  MAE:                  ${mae:.2f}")
print(f"  MAPE:                 {mape:.2f}%")
print(f"  R2 Score:             {r2:.4f}")
print(f"  Directional Accuracy: {dir_accuracy:.2f}%")
print(f"  Strategy Return:      {cumulative_strategy_return * 100:.2f}%")
print(f"  Buy & Hold Return:    {cumulative_buy_hold * 100:.2f}%")
print(f"  Sharpe Ratio:         {sharpe_ratio:.2f}")
print("=" * 50)

print(f"\n[7/7] Generating plot at {PLOT_SAVE_PATH}...")
os.makedirs(os.path.dirname(PLOT_SAVE_PATH), exist_ok=True)
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(y_test_unscaled, label='Actual NDX Close', color='black', linewidth=1.8)
plt.plot(y_pred_unscaled, label='MLP Predictions', color='#2563eb', linestyle='--', linewidth=1.5)
plt.title(f'MLP Predictions vs Actual (MAE: ${mae:.1f})', fontsize=12, fontweight='bold')
plt.xlabel('Timestep (Test Set Days)')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
strat_cum = np.cumprod(1 + strategy_returns)
bh_cum = np.cumprod(1 + actual_pct_change)
plt.plot(strat_cum, label=f'MLP Strategy (+{cumulative_strategy_return*100:.1f}%)', color='#10b981', linewidth=1.8)
plt.plot(bh_cum, label=f'Buy & Hold (+{cumulative_buy_hold*100:.1f}%)', color='#64748b', linestyle=':', linewidth=1.5)
plt.title(f'Backtest Equity Curve (Sharpe: {sharpe_ratio:.2f})', fontsize=12, fontweight='bold')
plt.xlabel('Trading Days')
plt.ylabel('Cumulative Multiple')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_SAVE_PATH, dpi=150)
plt.close()
print(f"  [OK] Saved results plot to {PLOT_SAVE_PATH}")
print("\n[DONE] MLP Model training and evaluation complete.")
