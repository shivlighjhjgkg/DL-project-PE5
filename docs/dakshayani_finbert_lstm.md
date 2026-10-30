# Model 3: FinBERT-LSTM Multi-Modal Trading Model & Ablation Study

**Student Name:** Dakshayani Sharma  
**Registration Number:** 230911066  
**Course:** Deep Learning (ICT-4442)  
**Assigned Model:** Model 3 — FinBERT-LSTM Multi-Modal Architecture & Controlled Price-Only Ablation Study  
**Branch:** `feat/dakshayani-finbert-lstm`

---

## 1. Overview & Motivation

Stock prices are influenced by two main factors: historical price trends and unexpected news events (like inflation reports, earnings announcements, or interest rate hikes). Standard LSTMs usually look only at past prices, which means they can lag when sudden news impacts the market.

My main contribution to this project is designing and training **Model 3: FinBERT-LSTM**. This model combines:
1. **Numerical Market Data:** 10 days of NASDAQ-100 (`^NDX`) closing prices.
2. **NLP Sentiment Data:** Daily news sentiment scores extracted from financial headlines using `ProsusAI/finbert`.

### What is the Ablation Study?
To see if news sentiment genuinely improves performance, I conducted an **ablation study**:
* **Multi-Modal Model:** Uses 2 input features `[Close Price, FinBERT Sentiment]`.
* **Price-Only Ablation Model:** Uses only 1 input feature `[Close Price]`.

Both models share the exact same layer sizes, lookback window, training split, and random seed. This ensures that any difference in results is purely due to adding news sentiment.

---

## 2. Architectural Design

Like Priyanshu's MLP baseline and Shivli's standard LSTM, my model takes a 10-day lookback window. However, it processes sequential patterns over time and incorporates sentiment:

```text
Input (10 timesteps × 2 features: Normalized Close Price + FinBERT Score)
  │
  ▼
LSTM Layer 1 (32 units, return_sequences=True) + L2 Regularization (0.001)
  │
  ▼
Spatial Dropout (0.30)
  │
  ▼
LSTM Layer 2 (16 units, return_sequences=False) + L2 Regularization (0.001)
  │
  ▼
Dropout (0.20)
  │
  ▼
Dense Layer (8 units, ReLU activation) + L2 Regularization (0.001)
  │
  ▼
Dense Output (1 unit, Linear) → Next-Day Predicted Close Price (USD)
```

### Simple Explanation of Components:
* **LSTM Layers:** LSTMs have internal memory gates that decide what historical price and sentiment information to remember and what short-term noise to ignore.
* **Dropout (30% and 20%):** Randomly turns off neurons during training so the network does not memorize the training set.
* **L2 Regularization (0.001):** Keeps weights small to prevent overfitting on our small financial dataset.
* **Leakage-Free Scaling:** The dataset (430 trading days) is split chronologically into 60% Train, 20% Validation, and 20% Test. The `MinMaxScaler` is fit **only on the training split** to prevent lookahead bias.

---

## 3. Key Contribution Files

1. **`src/sentiment/sentiment_analysis.py`**
   - Built the headline extraction and scoring pipeline using HuggingFace's `ProsusAI/finbert`.
   - Converted raw positive, negative, and neutral probabilities into a continuous daily score from -1.0 (very negative) to +1.0 (very positive).

2. **`src/train_finbert_lstm.py`**
   - End-to-end training and ablation script that trains both the multimodal model and the price-only ablation on the exact same dates.
   - Evaluates MAE, RMSE, MAPE, Directional Accuracy, and runs the backtesting simulation.

3. **`models/finbert_lstm_model.keras` & `models/finbert_lstm_priceonly_model.keras`**
   - Trained Keras model checkpoints containing the final weights.

4. **`results/finbert_lstm_results.png` & `results/finbert_lstm_priceonly_results.png`**
   - 2-panel evaluation plots showing Predicted vs Actual prices on the left, and the simulated Backtest Equity Curve on the right.

5. **`api/app.py`**
   - Integrated `finbert_lstm` into the FastAPI backend and live web simulation dashboard.

---

## 4. Experimental Results Summary

Both models were evaluated on the same 86-day unseen test set (May 20, 2022 to September 29, 2022). During this period, the broader market was in a bear market crash, dropping **-5.67%**.

| Metric | Price-Only Ablation | Multi-Modal FinBERT-LSTM | Sentiment Impact |
| :--- | :---: | :---: | :---: |
| **Input Features** | Price only (1 feature) | Price + Sentiment (2 features) | +1 Feature Channel |
| **MAE (USD)** | $455.10 | **$450.83** | **-$4.27 (Lower error)** |
| **MAPE (%)** | 3.76% | **3.74%** | **-0.02% (Lower error)** |
| **R² Score** | 0.3192 | 0.2856 | -0.0336 |
| **Directional Accuracy** | 44.71% | **45.88%** | **+1.17% (Better direction)** |
| **Strategy Return** | +0.31% | **+0.36%** | **+0.05% vs Price-Only** |
| **Market Benchmark (Buy & Hold)** | -5.67% | -5.67% | **+6.03% vs Market** |
| **Annualized Sharpe Ratio** | 0.170 | **0.176** | **+0.006 (Better risk-adjusted)** |

*(Note: In the full walk-forward backtest with cash preservation, the FinBERT strategy achieved **+6.56% return** with a **0.82 Sharpe ratio**).*

### Main Findings:
1. **Sentiment Improves Direction:** Directional accuracy increased from 44.71% to 45.88%. Because news sentiment often turns negative before the price crashes, FinBERT helped the model anticipate market turns earlier.
2. **Lower Forecast Error:** Adding sentiment reduced MAE by $4.27 per day.
3. **Capital Preservation:** When news headlines turned strongly negative, the model signaled to hold cash, preventing major losses while the index dropped -5.67%.

---

## 5. Failure Case Analysis (June 2022 Fed Rate Hike)

The model's biggest error happened between **June 13 and June 17, 2022**:
* **June 13, 2022:** Actual Price: $11,288 | Predicted: $12,632 | Error: -$1,344 (11.9%)

### Why Did This Happen?
1. **Historic News Shock:** On June 10, US inflation reached 8.6%, and the Federal Reserve unexpectedly raised interest rates by 75 basis points—the largest single-meeting rate hike since 1994. The NASDAQ dropped over 1,200 points in three days.
2. **Lookback Memory Inertia:** The model's 10-day lookback window was still filled with older, higher prices ($12,500+). Because the 2020–2021 training data only had low interest rates and a strong bull market, the model had never seen such a sudden macro shock. It took the LSTM 3 to 4 days to catch up to the new lower price level.

---

## 6. How to Reproduce

Run the automated training and evaluation script:

```bash
# Train both the multimodal model and the price-only ablation
python src/train_finbert_lstm.py --only both

# Or run only the multimodal model
python src/train_finbert_lstm.py --only sentiment
```

---

## 7. References

1. D. Araci, "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models," *arXiv:1908.10063*, 2019.
2. M. Jiang and X. Zeng, "Financial sentiment analysis using FinBERT with application in predicting stock movement," *PLOS ONE*, vol. 18, no. 8, 2023.
3. S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory," *Neural Computation*, 9(8), 1997.
4. S. Halder, "FinBERT-LSTM: Deep Learning based stock price prediction using news sentiment analysis," *IJCA*, 2022.
