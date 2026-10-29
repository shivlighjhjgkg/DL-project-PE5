"""
Unified Model Benchmark & Comparative Error Analysis
=====================================================
Evaluates all 4 project architectures + Ensemble under an identical protocol:
1. Model 1 (Priyanshu Singh): MLP Baseline
2. Model 2 (Shivli Dimri): Standard LSTM (Price-only)
3. Model 3 (Dakshayani Sharma): FinBERT-LSTM Fusion (Multimodal)
4. Model 4 (Prajika Saini): Advanced Attention-BiLSTM (Attention + Residual Skips)
5. Model 5: Blended Quantitative Ensemble

Generates:
- Consolidated comparison table across all models on common metrics
- In-depth error analysis and residual plots (fulfilling Report Section VI)
- Identification of failure cases on high-volatility news event days
- Exports results to results/all_models_comparison.json, csv, and png
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
import scipy.stats as stats

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

# Set local cache paths to avoid sandbox permission errors
os.environ.setdefault("KERAS_HOME", os.path.join(PROJECT_ROOT, ".keras"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(PROJECT_ROOT, ".matplotlib"))
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)

# Register Attention Layer for deserialization
@keras.utils.register_keras_serializable()
class TemporalAttentionLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(TemporalAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight', shape=(input_shape[-1], 1),
            initializer='glorot_uniform', trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias', shape=(input_shape[1], 1),
            initializer='zeros', trainable=True
        )
        super(TemporalAttentionLayer, self).build(input_shape)

    def call(self, x):
        e = tf.tanh(tf.matmul(x, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1)
        return tf.reduce_sum(x * a, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


STOCK_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'stock_price.csv')
SENTIMENT_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'sentiment_new.csv')
if not os.path.exists(SENTIMENT_PATH):
    SENTIMENT_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'sentiment.csv')

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
os.makedirs(RESULTS_DIR, exist_ok=True)

SEQUENCE_LENGTH = 10

def load_dataset():
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


def create_sequences(data_2d, seq_len=SEQUENCE_LENGTH):
    X, y = [], []
    for i in range(len(data_2d) - seq_len):
        X.append(data_2d[i:i + seq_len])
        y.append(data_2d[i + seq_len, 0])
    return np.array(X), np.array(y)


def compute_metrics(y_true, y_pred, model_name, owner_name, arch_desc):
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(mean_absolute_percentage_error(y_true, y_pred) * 100)
    r2 = float(r2_score(y_true, y_pred))

    pred_dir = np.diff(y_pred) > 0
    actual_dir = np.diff(y_true) > 0
    dir_acc = float(np.mean(pred_dir == actual_dir) * 100)

    # Trading simulation
    signals = np.where(y_pred[1:] > y_true[:-1], 1.0, 0.0)
    actual_pct = np.diff(y_true) / y_true[:-1]
    strategy_returns = signals * actual_pct
    strat_cum = float(np.prod(1 + strategy_returns) - 1)
    bh_cum = float(np.prod(1 + actual_pct) - 1)

    std = np.std(strategy_returns)
    sharpe = float((np.mean(strategy_returns) / std) * np.sqrt(252)) if std > 0 else 0.0
    win_rate = float(np.mean(strategy_returns[signals == 1.0] > 0) * 100) if np.sum(signals) > 0 else 0.0

    # Max Drawdown
    equity = np.cumprod(1 + strategy_returns)
    peak = np.maximum.accumulate(equity)
    drawdowns = (equity - peak) / peak
    mdd = float(np.min(drawdowns) * 100) if len(drawdowns) > 0 else 0.0

    return {
        'Model': model_name,
        'Member Owner': owner_name,
        'Architecture Summary': arch_desc,
        'MAE ($)': round(mae, 2),
        'RMSE ($)': round(rmse, 2),
        'MAPE (%)': round(mape, 2),
        'R²': round(r2, 4),
        'Direction Acc (%)': round(dir_acc, 2),
        'Strategy Return (%)': round(strat_cum * 100, 2),
        'Buy & Hold (%)': round(bh_cum * 100, 2),
        'Sharpe Ratio': round(sharpe, 2),
        'Win Rate (%)': round(win_rate, 2),
        'Max Drawdown (%)': round(mdd, 2)
    }


def main():
    print("=" * 80)
    print("UNIFIED MODEL BENCHMARKING & ERROR ANALYSIS")
    print("=" * 80)

    merged = load_dataset()
    dates = merged['Date'].values
    total_len = len(merged)
    train_size = int(total_len * 0.6)
    val_size = int(total_len * 0.2)

    train_df = merged.iloc[:train_size]
    val_df = merged.iloc[train_size:train_size + val_size]
    test_df = merged.iloc[train_size + val_size:]

    test_dates = dates[train_size + val_size + SEQUENCE_LENGTH:]

    # Scale Price (fit on train only)
    price_scaler = MinMaxScaler()
    train_price_s = price_scaler.fit_transform(train_df[['Close']])
    val_price_s = price_scaler.transform(val_df[['Close']])
    test_price_s = price_scaler.transform(test_df[['Close']])

    # Scale Sentiment (fit on train only)
    sent_scaler = MinMaxScaler()
    train_sent_s = sent_scaler.fit_transform(train_df[['Sentiment']])
    val_sent_s = sent_scaler.transform(val_df[['Sentiment']])
    test_sent_s = sent_scaler.transform(test_df[['Sentiment']])

    # Test sequences
    test_2d = np.hstack([test_price_s, test_sent_s])
    X_test_2feat, y_test_s = create_sequences(test_2d)
    X_test_1feat = X_test_2feat[:, :, 0:1]

    y_test_unscaled = price_scaler.inverse_transform(y_test_s.reshape(-1, 1)).flatten()

    print(f"Test samples: {len(y_test_unscaled)} trading days [{test_dates[0]} to {test_dates[-1]}]")

    model_registry = [
        {
            'name': 'MLP Baseline',
            'owner': 'Priyanshu Singh (230911412)',
            'desc': 'Dense 64->32->16->1 (ReLU, L2, Dropout)',
            'path': os.path.join(MODELS_DIR, 'mlp_model.keras'),
            'input_type': '2feat',
            'custom_objects': None
        },
        {
            'name': 'Standard LSTM',
            'owner': 'Shivli Dimri (230953114)',
            'desc': 'Stacked LSTM 32->16 (Price-only, L2, Dropout)',
            'path': os.path.join(MODELS_DIR, 'lstm_model.keras'),
            'input_type': '1feat',
            'custom_objects': None
        },
        {
            'name': 'FinBERT-LSTM',
            'owner': 'Dakshayani Sharma (230911066)',
            'desc': 'Multimodal LSTM 32->16 + Sentiment input',
            'path': os.path.join(MODELS_DIR, 'finbert_lstm_model.keras'),
            'input_type': '2feat',
            'custom_objects': None
        },
        {
            'name': 'Attention-BiLSTM',
            'owner': 'Prajika Saini (230953106)',
            'desc': 'BiLSTM 32->16 + Attention + Residual Skips',
            'path': os.path.join(MODELS_DIR, 'attention_bilstm_model.keras'),
            'input_type': '2feat',
            'custom_objects': {'TemporalAttentionLayer': TemporalAttentionLayer}
        }
    ]

    predictions_dict = {}
    table_rows = []

    for m_info in model_registry:
        print(f"\nEvaluating: {m_info['name']}...")
        if not os.path.exists(m_info['path']):
            print(f"  [WARN] Model file not found at {m_info['path']}, skipping.")
            continue

        try:
            model = keras.models.load_model(m_info['path'], custom_objects=m_info['custom_objects'])
            inp = X_test_2feat if m_info['input_type'] == '2feat' else X_test_1feat
            pred_s = model.predict(inp, verbose=0).flatten()
            pred_unscaled = price_scaler.inverse_transform(pred_s.reshape(-1, 1)).flatten()
            predictions_dict[m_info['name']] = pred_unscaled

            row = compute_metrics(
                y_test_unscaled, pred_unscaled,
                m_info['name'], m_info['owner'], m_info['desc']
            )
            table_rows.append(row)
            print(f"  MAE: ${row['MAE ($)']:.2f} | R²: {row['R²']:.4f} | Dir Acc: {row['Direction Acc (%)']}% | Sharpe: {row['Sharpe Ratio']}")
        except Exception as e:
            print(f"  [ERROR] Failed to evaluate {m_info['name']}: {e}")

    # Build Ensemble if models exist
    if 'Attention-BiLSTM' in predictions_dict and 'FinBERT-LSTM' in predictions_dict and 'MLP Baseline' in predictions_dict:
        print("\nEvaluating: Quantitative Ensemble...")
        # Optimal blended weighting: 45% Attention-BiLSTM + 35% FinBERT-LSTM + 20% MLP
        ens_pred = (
            0.45 * predictions_dict['Attention-BiLSTM'] +
            0.35 * predictions_dict['FinBERT-LSTM'] +
            0.20 * predictions_dict['MLP Baseline']
        )
        predictions_dict['Ensemble (Blended)'] = ens_pred
        ens_row = compute_metrics(
            y_test_unscaled, ens_pred,
            'Quantitative Ensemble', 'Joint Team Model',
            'Weighted Blend (45% Attn-BiLSTM, 35% FinBERT-LSTM, 20% MLP)'
        )
        table_rows.append(ens_row)
        print(f"  MAE: ${ens_row['MAE ($)']:.2f} | R²: {ens_row['R²']:.4f} | Dir Acc: {ens_row['Direction Acc (%)']}% | Sharpe: {ens_row['Sharpe Ratio']}")

    # Save Comparison Data
    df_results = pd.DataFrame(table_rows)
    csv_path = os.path.join(RESULTS_DIR, 'all_models_comparison.csv')
    df_results.to_csv(csv_path, index=False)
    print(f"\n[OK] Saved results CSV to {csv_path}")

    json_path = os.path.join(RESULTS_DIR, 'all_models_comparison.json')
    with open(json_path, 'w') as f:
        json.dump(table_rows, f, indent=2)
    print(f"[OK] Saved results JSON to {json_path}")

    # Print Table
    print("\n" + "=" * 110)
    print("FINAL MODEL COMPARISON TABLE (Common Protocol: 60/20/20 Chronological Split, 10-day lookback)")
    print("=" * 110)
    cols_display = ['Model', 'MAE ($)', 'RMSE ($)', 'MAPE (%)', 'R²', 'Direction Acc (%)', 'Strategy Return (%)', 'Sharpe Ratio', 'Win Rate (%)']
    print(df_results[cols_display].to_string(index=False))
    print("=" * 110)

    # Render Table Graphic
    fig, ax = plt.subplots(figsize=(13, len(table_rows) * 0.9 + 1.2))
    ax.axis('tight')
    ax.axis('off')
    col_labels = ['Model', 'MAE ($)', 'RMSE ($)', 'MAPE (%)', 'R²', 'Dir Acc (%)', 'Return (%)', 'Sharpe', 'Win Rate (%)']
    cell_data = df_results[['Model', 'MAE ($)', 'RMSE ($)', 'MAPE (%)', 'R²', 'Direction Acc (%)', 'Strategy Return (%)', 'Sharpe Ratio', 'Win Rate (%)']].values
    table = ax.table(cellText=cell_data, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor('#1e293b')
            cell.set_text_props(color='white', weight='bold')
        elif i % 2 == 0:
            cell.set_facecolor('#f1f5f9')
    table_img_path = os.path.join(RESULTS_DIR, 'comparison_table.png')
    plt.savefig(table_img_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved comparison table image to {table_img_path}")

    # ========================================================================
    # ERROR ANALYSIS & FAILURE CASE INVESTIGATION (Section VI)
    # ========================================================================
    print("\nGenerating comprehensive error analysis plots...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    colors = {
        'MLP Baseline': '#2563eb',
        'Standard LSTM': '#dc2626',
        'FinBERT-LSTM': '#059669',
        'Attention-BiLSTM': '#7c3aed',
        'Quantitative Ensemble': '#d97706'
    }

    # Plot 1: Actual vs Predictions Over Time
    ax1 = axes[0, 0]
    ax1.plot(test_dates, y_test_unscaled, label='Actual NDX Close', color='black', linewidth=2.0)
    for name, pred in predictions_dict.items():
        c = colors.get(name, '#64748b')
        ax1.plot(test_dates, pred, label=name, color=c, linestyle='--', alpha=0.85, linewidth=1.3)
    ax1.set_title('Test Set Predictions vs Actual NASDAQ-100 Price', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price (USD)')
    ax1.legend(fontsize=8, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=30)

    # Plot 2: Residual Distribution (Error Histogram + KDE)
    ax2 = axes[0, 1]
    for name, pred in predictions_dict.items():
        residuals = y_test_unscaled - pred
        sns.kdeplot(residuals, ax=ax2, label=f"{name} (μ={np.mean(residuals):.1f})",
                    color=colors.get(name, '#64748b'), linewidth=1.8)
    ax2.axvline(0, color='gray', linestyle=':', linewidth=1.5)
    ax2.set_title('Prediction Error Distribution (Residuals = Actual - Pred)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Residual ($)')
    ax2.set_ylabel('Density')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Q-Q Plot for Best Model (Attention-BiLSTM)
    ax3 = axes[1, 0]
    best_name = 'Attention-BiLSTM' if 'Attention-BiLSTM' in predictions_dict else list(predictions_dict.keys())[0]
    best_res = y_test_unscaled - predictions_dict[best_name]
    stats.probplot(best_res, dist="norm", plot=ax3)
    ax3.get_lines()[0].set_markerfacecolor('#7c3aed')
    ax3.get_lines()[0].set_markeredgecolor('#7c3aed')
    ax3.get_lines()[1].set_color('black')
    ax3.set_title(f'Normal Q-Q Plot of Residuals ({best_name})', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Absolute Error vs Market Volatility (Failure Case Analysis)
    ax4 = axes[1, 1]
    daily_vol = np.abs(np.diff(y_test_unscaled) / y_test_unscaled[:-1]) * 100
    for name in [best_name, 'MLP Baseline']:
        if name in predictions_dict:
            abs_err = np.abs(y_test_unscaled[1:] - predictions_dict[name][1:])
            ax4.scatter(daily_vol, abs_err, alpha=0.7, label=f"{name} Errors", color=colors.get(name, 'blue'))
    ax4.set_title('Absolute Prediction Error vs Daily Price Volatility (%)', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Actual Daily Price Change (%)')
    ax4.set_ylabel('Absolute Error (USD)')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    error_plot_path = os.path.join(RESULTS_DIR, 'error_analysis.png')
    plt.savefig(error_plot_path, dpi=180)
    plt.close()
    print(f"[OK] Saved error analysis plot to {error_plot_path}")

    # Save Failure Cases Report
    worst_idx = np.argsort(np.abs(best_res))[-5:][::-1]
    failure_cases = []
    for idx in worst_idx:
        failure_cases.append({
            'date': str(pd.to_datetime(test_dates[idx]).date()),
            'actual_price': float(y_test_unscaled[idx]),
            'predicted_price': float(predictions_dict[best_name][idx]),
            'error_usd': float(best_res[idx]),
            'error_pct': float(abs(best_res[idx]) / y_test_unscaled[idx] * 100)
        })
    failure_path = os.path.join(RESULTS_DIR, 'failure_cases.json')
    with open(failure_path, 'w') as f:
        json.dump(failure_cases, f, indent=2)
    print(f"[OK] Saved failure case details to {failure_path}")
    print("\n[DONE] Unified evaluation and error analysis complete.")


if __name__ == '__main__':
    main()
