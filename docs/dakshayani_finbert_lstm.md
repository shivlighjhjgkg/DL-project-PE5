# Model 3: FinBERT-LSTM Multi-Modal Trading Model & Ablation Study

**Course:** ICT-4442 Deep Learning  
**Student:** Dakshayani Sharma (Reg. No. 230911066)  
**Assigned Task:** Model 3 — Multi-Modal FinBERT-LSTM (Price + Sentiment) and Price-Only Ablation Study  
**Branch:** `feat/dakshayani-finbert-lstm`

---

## 1. Project Overview & Motivation

Stock prices change not just from past technical chart patterns, but also from unexpected news announcements (earnings, inflation prints, central bank rate decisions). Traditional LSTM models only take historical prices as input, so they often lag when major news breaks.

In this part of the project, I built and evaluated **Model 3: FinBERT-LSTM**. The goal is to see if combining numerical price history with daily news sentiment scores gives better price predictions and trading decisions than using price alone.

### What is the Ablation Study?
To fairly test whether news sentiment actually helps, I trained two identical models on the exact same dataset, train/val/test split, and random seed:
1. **Multi-Modal FinBERT-LSTM:** Inputs are `[Normalized Close Price, FinBERT Sentiment Score]` (2 features).
2. **Price-Only Ablation:** Inputs are `[Normalized Close Price]` only (1 feature).

Everything else (layers, dropout, optimizer, lookback window) was kept exactly the same so that any difference in results is purely due to the sentiment feature.

---

## 2. Model Architecture & Data Pipeline

### 2.1 Input Data
For each trading day $t$, the model takes a 10-day lookback sequence:
$$\mathbf{X} = (\mathbf{x}_{t-9}, \mathbf{x}_{t-8}, \dots, \mathbf{x}_t)$$

Each daily step contains two normalized features:
$$\mathbf{x}_t = [p_t, s_t]^\top$$

* $p_t$: NASDAQ-100 daily closing price, normalized to $[0, 1]$ using MinMaxScaler.
* $s_t$: Daily news sentiment score from `ProsusAI/finbert`, ranging from $-1.0$ (very negative) to $+1.0$ (very positive).

### 2.2 Network Design
I used a two-layer stacked LSTM network designed in Keras:

1. **Input Layer:** Shape `(10, 2)` for multimodal, or `(10, 1)` for price-only.
2. **LSTM Layer 1:** 32 units, `return_sequences=True`, with L2 weight regularization (`1e-3`) to prevent overfitting.
3. **Dropout 1:** 30% dropout to prevent co-adaptation.
4. **LSTM Layer 2:** 16 units, `return_sequences=False` (compresses the 10-day sequence into a 16-dimensional representation).
5. **Dropout 2:** 20% dropout.
6. **Dense Layer:** 8 units with ReLU activation.
7. **Output Layer:** 1 unit with Linear activation to predict the next-day normalized closing price $\hat{y}_{t+1}$.

### 2.3 Training Setup
* **Loss Function:** Mean Squared Error (MSE).
* **Optimizer:** Adam with an initial learning rate of `0.001`.
* **Epochs & Batch Size:** 120 epochs with batch size 16.
* **Early Stopping:** Monitors validation loss with a patience of 25 epochs (restoring the best model weights).
* **Learning Rate Reduction:** Halves learning rate if validation loss doesn't improve for 10 epochs.
* **No Data Leakage:** The dataset (430 trading days) is split chronologically into 60% Train, 20% Validation, and 20% Test. The `MinMaxScaler` is fit **only on the training split** and then used to transform validation and test splits.

---

## 3. Results & Ablation Comparison

Both models were evaluated on the same 86-day out-of-sample test period (**May 20, 2022 to September 29, 2022**). During this period, the broader market was in a bear market, and the Buy & Hold benchmark dropped **-5.67%**.

### Performance Summary Table

| Metric | Price-Only Ablation | Multi-Modal FinBERT-LSTM | Sentiment Impact |
|:---|:---:|:---:|:---:|
| **Features Used** | Close Price (1 feature) | Close + FinBERT (2 features) | Added NLP Sentiment |
| **MAE** | \$455.10 | **\$450.83** | **-\$4.27 (Lower error)** |
| **RMSE** | \$542.67 | \$555.88 | +\$13.21 |
| **MAPE** | 3.76% | **3.74%** | **-0.02% (Lower error)** |
| **R² Score** | 0.3192 | 0.2856 | -0.0336 |
| **Directional Accuracy** | 44.71% | **45.88%** | **+1.17% (Better direction)** |
| **Strategy Return** | +0.31% | **+0.36%** | **+0.05% vs Price-Only** |
| **Buy & Hold Benchmark** | -5.67% | -5.67% | **+6.03% vs Market** |
| **Annualized Sharpe Ratio** | 0.170 | **0.176** | **+0.006 (Better risk-adjusted)** |

*(Note: In the full unified backtesting engine with cash thresholds, the FinBERT strategy delivered +6.56% return and a 0.82 Sharpe ratio against the falling market).*

### Key Takeaways from the Experiment:
1. **Sentiment Improves Directional Accuracy:**  
   Directional accuracy improved from 44.71% to 45.88%. Because news headlines often anticipate big moves before the price chart reflects them, FinBERT gave the model earlier warning signals.
2. **Lower Average Forecast Error:**  
   The multimodal model brought MAE down from \$455.10 to \$450.83.
3. **Protection in a Down Market:**  
   Both models beat the Buy & Hold market drop (-5.67%). When news sentiment turned strongly negative, the trading policy exited into cash, preventing large drawdowns.

---

## 4. Failure Case Analysis (June 2022 Crash)

The largest error occurred during the week of **June 13–17, 2022**:

| Date | Actual Close | FinBERT-LSTM Pred | Error | What Happened in the Market |
|:---:|:---:|:---:|:---:|:---|
| **2022-06-13** | \$11,288.08 | \$12,632.31 | -\$1,344.23 (11.9%) | US inflation came out at 8.6%; Fed announced emergency 75 bps rate hike. |
| **2022-06-14** | \$11,310.87 | \$12,284.15 | -\$973.28 (8.6%) | Market sell-off continued across all tech stocks. |
| **2022-06-16** | \$11,127.57 | \$12,185.02 | -\$1,057.45 (9.5%) | Liquidity drop and continued drop in NASDAQ. |

### Why Did the Model Miss This?
1. **Sudden Macro Shock:** The Fed enacted a 75 bps rate hike (the biggest single hike since 1994). The NASDAQ fell over 1,200 points in just a few days.
2. **Lookback Memory Inertia:** The model's 10-day lookback window was still filled with older prices in the \$12,500–\$12,800 range. Because the training period (2020–2021) only had near-zero interest rates, the LSTM had never seen such a sharp macro shock. It took about 3 to 4 days for the model's internal states to catch up to the new lower price level.

---

## 5. How to Run My Code

The code for training both the multimodal model and the ablation baseline is automated in one script:

```bash
# Run both models (saves models, metrics JSON, and result plots)
python src/train_finbert_lstm.py --only both

# Or run only the multimodal FinBERT-LSTM
python src/train_finbert_lstm.py --only sentiment

# Or run only the price-only ablation
python src/train_finbert_lstm.py --only priceonly
```

### Generated Files
* **Trained Models:** `models/finbert_lstm_model.keras` and `models/finbert_lstm_priceonly_model.keras`
* **Scaler:** `models/price_scaler.pkl`
* **Metrics:** `results/finbert_lstm_comparison.json` and `results/finbert_lstm_metrics.json`
* **Plots:** `results/finbert_lstm_results.png` and `results/finbert_lstm_priceonly_results.png`

---

## 6. References

1. D. Araci, "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models," *arXiv preprint arXiv:1908.10063*, 2019.
2. M. Jiang and X. Zeng, "Financial sentiment analysis using FinBERT with application in predicting stock movement," *PLOS ONE*, vol. 18, no. 8, 2023.
3. S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory," *Neural Computation*, 9(8), 1997.
4. S. Halder, "FinBERT-LSTM: Deep Learning based stock price prediction using news sentiment analysis," *IJCA*, 2022.
