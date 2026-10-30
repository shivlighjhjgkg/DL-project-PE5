# Model 1: Multilayer Perceptron (MLP) Baseline Model

**Course:** ICT-4442 Deep Learning  
**Student:** Priyanshu Singh (Reg. No. 230911412)  
**Assigned Task:** Model 1 — Non-Recurrent Multi-Layer Perceptron (MLP) Baseline with Dense Feature Interaction & Ensembling  
**Branch:** `feat/priyanshu`

---

## 1. Overview & Motivation

In algorithmic trading with deep learning, recurrent architectures (such as LSTMs and GRUs) are often assumed to be strictly necessary. However, recurrent cells can suffer from sequence memory inertia, slow convergence, and severe overfitting when trained on small financial datasets (e.g., under 500 trading days).

To establish a rigorous, transparent benchmark for our project, I designed and evaluated **Model 1: Multilayer Perceptron (MLP) Baseline**. The objectives are:
1. **Non-Recurrent Baseline:** Provide a fast, feedforward benchmark that maps historical price and sentiment features directly to the next day's price without recurrent hidden-state persistence.
2. **Dense Feature Interaction:** Flatten the multi-modal lookback window to allow dense layers to freely capture cross-day and cross-feature interactions.
3. **Anchor for Joint Ensembling:** Contribute a structurally diverse model to the project's final weighted ensemble ($0.20 \times \text{MLP} + 0.35 \times \text{FinBERT-LSTM} + 0.45 \times \text{Attention-BiLSTM}$).

---

## 2. Model Architecture & Training Pipeline

### 2.1 Input Representation & Preprocessing
* **Lookback Window:** $T = 10$ trading days.
* **Features per Step:** $D = 2$ features:
  $$\mathbf{x}_t = [p_t, s_t]^\top$$
  - $p_t$: Normalized daily NASDAQ-100 closing price (`MinMaxScaler` fitted strictly on training data).
  - $s_t$: Daily FinBERT sentiment polarity score from financial news headlines.
* **Sequence Input Shape:** `(10, 2)` $\to$ Flattened into a 20-dimensional feature vector.

### 2.2 Network Architecture
Implemented in [`src/train_mlp.py`](../src/train_mlp.py):

```text
Input Sequence: (10 timesteps, 2 features)
               │
               ▼
Flatten Layer: 20-dimensional input vector
               │
               ▼
Dense Layer 1: 64 units, ReLU, L2 Regularization (λ = 1e-3)
               │
               ▼
Dropout Layer 1: 30% dropout rate
               │
               ▼
Dense Layer 2: 32 units, ReLU, L2 Regularization (λ = 1e-3)
               │
               ▼
Dropout Layer 2: 20% dropout rate
               │
               ▼
Dense Layer 3: 16 units, ReLU, L2 Regularization (λ = 1e-3)
               │
               ▼
Dense Output: 1 unit, Linear activation -> Next-Day Close Price (USD)
```

### 2.3 Training & Regularization Hyperparameters
* **Loss Function:** Mean Squared Error (MSE).
* **Optimizer:** Adam ($\text{learning rate} = 0.001$).
* **Batch Size & Epochs:** Batch size of 16, maximum 120 epochs.
* **Callbacks:**
  - `EarlyStopping`: Patience = 25 epochs (restoring best validation weights).
  - `ReduceLROnPlateau`: Halves learning rate if validation loss plateaus for 10 epochs.
* **Strict Leakage Prevention:** 60% Train (288 days), 20% Validation (96 days), 20% Test (96 days). The scaler is fitted exclusively on the training split.

---

## 3. Empirical Results & Performance Analysis

All models were evaluated on the exact same 86-day out-of-sample test window (**May 20, 2022 to September 29, 2022**), during which tech equities suffered a sharp macro downturn (NASDAQ-100 Buy & Hold benchmark dropped **-5.67%**).

### Performance Summary Table

| Metric | Baseline MLP (Model 1) | Standard LSTM | FinBERT-LSTM | Buy & Hold Benchmark |
| :--- | :---: | :---: | :---: | :---: |
| **Model Type** | Feedforward MLP | Recurrent (Price-Only) | Recurrent (Price + Sentiment) | Market Index |
| **MAE ($)** | **$350.84** | $428.17 | $450.83 | — |
| **RMSE ($)** | **$413.70** | $523.85 | $555.88 | — |
| **MAPE (%)** | **2.87%** | 3.54% | 3.74% | — |
| **$R^2$ Score** | **0.6043** | 0.3656 | 0.2856 | — |
| **Directional Accuracy** | **48.24%** | 43.53% | 45.88% | — |
| **Strategy Return (%)** | **+6.10%** | -1.52% | +0.36% | -5.67% |
| **Sharpe Ratio** | **0.84** | -0.03 | 0.18 | -0.21 |
| **Max Drawdown (%)** | **-9.58%** | -11.81% | -11.81% | -17.82% |
| **Win Rate (%)** | **54.55%** | 52.08% | 52.08% | — |

### Key Findings
1. **Superior Regression Accuracy:** The regularized MLP achieved an MAE of **$350.84** and an $R^2$ of **0.6043**, outperforming both standard and multi-modal LSTMs on raw regression error.
2. **Defensive Capital Preservation:** In a declining market (-5.67%), the MLP trading policy delivered **+6.10% net return** (an excess return of **+11.77%** over the market benchmark) and had the lowest maximum drawdown (**-9.58%**).
3. **No Recurrent Memory Lag:** Because the MLP does not carry forward hidden states from previous time steps, it adjusted more rapidly to regime shifts compared to standard recurrent units.

---

## 4. Failure Mode & Boundary Analysis

* **Unseen Regime Shocks:** Like the recurrent baselines, the MLP exhibited its largest residual errors during the June 2022 Fed rate hike shock (June 13–16, 2022).
* **Flat Temporal Weighting:** Because the input window is flattened, the MLP must learn position-specific weights from scratch without the inherent inductive bias of sequential ordering, making proper L2 and Dropout regularization critical to prevent memorization.

---

## 5. Execution & Artifacts

### Running the Training Pipeline
```bash
# Train the MLP baseline, evaluate test metrics, and export checkpoints
python src/train_mlp.py
```

### Generated Files & Artifacts
* **Model Checkpoint:** `models/mlp_model.keras`
* **API Checkpoint:** `api/mlp_model.keras`
* **Metrics Export:** `results/mlp_metrics.json`
* **Evaluation Plot:** `results/mlp_model_results.png`

---

## 6. References

1. S. Halder, "FinBERT-LSTM: Deep Learning based stock price prediction using News Sentiment Analysis," *IJCA*, 2022.
2. F. Feng et al., "Temporal Relational Ranking for Stock Prediction," *ACM TOIS*, 2019.
3. N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, "Dropout: A simple way to prevent neural networks from overfitting," *JMLR*, 2014.
4. D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in *Proc. ICLR*, 2015.
