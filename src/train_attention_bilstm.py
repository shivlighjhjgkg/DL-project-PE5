import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler

# 1. Output directories
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# 2. Custom Temporal Attention Layer
@tf.keras.utils.register_keras_serializable()
class TemporalAttentionLayer(layers.Layer):
    """
    Computes temporal attention: alpha_t = softmax(v^T * tanh(W * h_t + b))
    """
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

# 3. Load & Clean Stock Price Data
price_raw = pd.read_csv("data/raw/stock_price.csv")

# Skip metadata rows (0: Ticker, 1: Date)
price_clean = price_raw.iloc[2:].copy()
price_clean.columns = [c.strip() for c in price_raw.columns]
price_clean = price_clean.rename(columns={"Price": "Date"})
price_clean["Date"] = pd.to_datetime(price_clean["Date"])
price_clean["Close"] = pd.to_numeric(price_clean["Close"], errors="coerce")
price_clean = price_clean.dropna(subset=["Date", "Close"])[["Date", "Close"]]

# Load & Clean Sentiment Data
sent_raw = pd.read_csv("data/processed/sentiment_new.csv")
sent_raw.columns = [c.strip() for c in sent_raw.columns]
sent_date_col = next((c for c in sent_raw.columns if c.lower() == "date"), sent_raw.columns[0])
sent_raw["Date"] = pd.to_datetime(sent_raw[sent_date_col])

sentiment_col = next((c for c in sent_raw.columns if any(k in c.lower() for k in ["finbert", "sentiment", "score", "compound"])), sent_raw.columns[-1])
sent_clean = sent_raw[["Date", sentiment_col]].copy()
sent_clean[sentiment_col] = pd.to_numeric(sent_clean[sentiment_col], errors="coerce").fillna(0.0)

# Merge Price and Sentiment on Date
df = pd.merge(price_clean, sent_clean, on="Date", how="inner").sort_values("Date").reset_index(drop=True)
print(f"Data successfully aligned: {len(df)} records.")

features = df[["Close", sentiment_col]].values
dates = df["Date"].values

# 4. Chronological 60/20/20 Split
n = len(features)
train_size = int(n * 0.60)
val_size = int(n * 0.20)

train_raw = features[:train_size]
val_raw = features[train_size:train_size + val_size]
test_raw = features[train_size + val_size:]

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_raw)
val_scaled = scaler.transform(val_raw)
test_scaled = scaler.transform(test_raw)

scaled_all = np.vstack([train_scaled, val_scaled, test_scaled])

# 5. Create 10-day lookback windows
LOOKBACK = 10
X, y = [], []
for i in range(LOOKBACK, len(scaled_all)):
    X.append(scaled_all[i - LOOKBACK:i])
    y.append(scaled_all[i, 0])

X = np.array(X)
y = np.array(y)

train_idx = train_size - LOOKBACK
val_idx = (train_size + val_size) - LOOKBACK

X_train, y_train = X[:train_idx], y[:train_idx]
X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
X_test, y_test = X[val_idx:], y[val_idx:]
test_dates = dates[LOOKBACK + val_idx:]

# 6. Model Architecture: Bidirectional LSTM + Residual Skip + Attention
inputs = layers.Input(shape=(LOOKBACK, X.shape[2]))
bilstm_out = layers.Bidirectional(layers.LSTM(32, return_sequences=True), name="bi_lstm")(inputs)
skip_proj = layers.Dense(64, name="residual_projection")(inputs)
residual_sum = layers.Add()([bilstm_out, skip_proj])

context_vector, attention_weights = TemporalAttentionLayer(name="temporal_attention")(residual_sum)
dense_out = layers.Dense(32, activation="relu")(context_vector)
dense_out = layers.Dropout(0.2)(dense_out)
outputs = layers.Dense(1, name="price_prediction")(dense_out)

model = models.Model(inputs=inputs, outputs=outputs)
attention_extractor = models.Model(inputs=inputs, outputs=attention_weights)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")

early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)

print("Training Attention-BiLSTM model...")
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)

# 7. Save Checkpoint
model.save("models/attention_bilstm_model.keras")
print("Saved models/attention_bilstm_model.keras")

# 8. Unscale Predictions and Standard Metrics
y_pred_scaled = model.predict(X_test).flatten()
close_min = scaler.data_min_[0]
close_max = scaler.data_max_[0]

y_test_real = y_test * (close_max - close_min) + close_min
y_pred_real = y_pred_scaled * (close_max - close_min) + close_min

mae = float(np.mean(np.abs(y_test_real - y_pred_real)))
rmse = float(np.sqrt(np.mean((y_test_real - y_pred_real) ** 2)))
mape = float(np.mean(np.abs((y_test_real - y_pred_real) / y_test_real)) * 100)

# Directional Accuracy
actual_returns = (y_test_real[1:] - y_test_real[:-1]) / y_test_real[:-1]
pred_direction = np.where(y_pred_real[1:] > y_test_real[:-1], 1, -1)
actual_direction = np.where(actual_returns > 0, 1, -1)
directional_acc = float(np.mean(pred_direction == actual_direction) * 100)

# 9. Walk-Forward Backtesting Simulation (Matching Team Specs)
# Strategy: Long (1) if model predicts price will rise, Short/Cash (0 or -1) otherwise
signals = np.where(y_pred_real[1:] > y_test_real[:-1], 1.0, 0.0)
strategy_returns = signals * actual_returns

# Cumulative Equity Curves
strategy_equity = np.cumprod(1 + strategy_returns)
buy_and_hold_equity = np.cumprod(1 + actual_returns)

# Prepend starting multiple of 1.0
strategy_equity = np.insert(strategy_equity, 0, 1.0)
buy_and_hold_equity = np.insert(buy_and_hold_equity, 0, 1.0)

strategy_total_return = float((strategy_equity[-1] - 1.0) * 100)
bh_total_return = float((buy_and_hold_equity[-1] - 1.0) * 100)

# Annualized Sharpe Ratio (Rf = 0%)
excess_returns = strategy_returns
sharpe_ratio = float(np.mean(excess_returns) / (np.std(excess_returns) + 1e-9) * np.sqrt(252))

metrics = {
    "model": "Attention-BiLSTM",
    "MAE": round(mae, 2),
    "RMSE": round(rmse, 2),
    "MAPE": round(mape, 2),
    "directional_accuracy_pct": round(directional_acc, 2),
    "strategy_return_pct": round(strategy_total_return, 2),
    "buy_and_hold_return_pct": round(bh_total_return, 2),
    "sharpe_ratio": round(sharpe_ratio, 2)
}

with open("results/attention_bilstm_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
print("Updated Metrics saved:", metrics)

# 10. Two-Subplot Results Figure (Prediction + Equity Curve)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Left Subplot: Predictions vs Actual
ax1.plot(y_test_real, label="Actual NDX Close", color="black", linewidth=1.2)
ax1.plot(y_pred_real, label="Attention-BiLSTM Predictions", color="firebrick", linestyle="--", linewidth=1.2)
ax1.set_title(f"Attention-BiLSTM Predictions vs Actual (MAE: ${mae:.1f})")
ax1.set_xlabel("Timestep (Test Set Days)")
ax1.set_ylabel("Price (USD)")
ax1.legend(loc="upper right", frameon=True)
ax1.grid(True, linestyle="--", alpha=0.5)

# Right Subplot: Backtest Equity Curve
ax2.plot(strategy_equity, label=f"Attention-BiLSTM Strategy ({strategy_total_return:+.1f}%)", color="mediumseagreen", linewidth=1.4)
ax2.plot(buy_and_hold_equity, label=f"Buy & Hold ({bh_total_return:+.1f}%)", color="silver", linestyle=":", linewidth=1.2)
ax2.set_title(f"Backtest Equity Curve (Sharpe: {sharpe_ratio:.2f})")
ax2.set_xlabel("Trading Days")
ax2.set_ylabel("Cumulative Multiple")
ax2.legend(loc="upper left", frameon=True)
ax2.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("results/attention_bilstm_results.png", dpi=200)
plt.close()
print("Saved 2-panel comparison figure to results/attention_bilstm_results.png")

# 11. Attention Heatmap Plot
att_weights = attention_extractor.predict(X_test).squeeze()
sample_weights = att_weights[-15:]

plt.figure(figsize=(10, 6))
plt.imshow(sample_weights, aspect="auto", cmap="Blues")
plt.colorbar(label="Attention Weight")
plt.title("Temporal Attention Weights Across 10 Lookback Steps (Recent Test Days)")
plt.xlabel("Lookback Step (t-10 to t-1)")
plt.ylabel("Test Day Index")
plt.xticks(range(10), [f"t-{10-i}" for i in range(10)])
plt.tight_layout()
plt.savefig("results/attention_weights.png", dpi=200)
plt.close()

print("All tasks completed successfully!")