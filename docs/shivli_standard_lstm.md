# Model 2: Standard Price-Only LSTM Baseline

**Course:** Deep Learning (ICT-4442)  
**Department:** Information and Communication Technology, Manipal Institute of Technology  
**Model Owner:** Shivli Dimri (Registration No. 230953114)  
**Assigned Deliverable:** Standard sequential price-only LSTM baseline

---

## 1. Purpose and Summary

The Standard LSTM is the project's sequential price-only baseline. It uses
historical NDX closing prices to predict the next closing price without news,
FinBERT sentiment, attention, or bidirectional processing. This isolates the
value of a plain recurrent architecture and provides a consistent comparison
point for the MLP, FinBERT-LSTM, and Attention-BiLSTM models.

The implementation is in [`src/train_lstm.py`](../src/train_lstm.py). It uses
10-day rolling windows, chronological data splits, train-only scaling, and the
same core regression metrics used by the other baseline models.

## 2. Mathematical Formulation

Let \(p_t\) denote the NDX closing price on trading day \(t\). The input to the
model is a normalized price sequence of length \(T=10\):

$$
\mathbf{X}_t = (p_{t-9}, p_{t-8}, \ldots, p_t) \in \mathbb{R}^{10 \times 1}
$$

The model predicts the next normalized closing price:

$$
\hat{p}_{t+1} = f_\theta(\mathbf{X}_t)
$$

For each LSTM unit, the hidden state \(\mathbf{h}_t\) and cell state
\(\mathbf{C}_t\) are updated using four gates:

$$
\mathbf{f}_t =
\sigma(\mathbf{W}_f\mathbf{x}_t +
\mathbf{U}_f\mathbf{h}_{t-1} + \mathbf{b}_f)
\quad \text{(forget gate)}
$$

$$
\mathbf{i}_t =
\sigma(\mathbf{W}_i\mathbf{x}_t +
\mathbf{U}_i\mathbf{h}_{t-1} + \mathbf{b}_i)
\quad \text{(input gate)}
$$

$$
\tilde{\mathbf{C}}_t =
\tanh(\mathbf{W}_c\mathbf{x}_t +
\mathbf{U}_c\mathbf{h}_{t-1} + \mathbf{b}_c)
\quad \text{(candidate state)}
$$

$$
\mathbf{C}_t =
\mathbf{f}_t \odot \mathbf{C}_{t-1} +
\mathbf{i}_t \odot \tilde{\mathbf{C}}_t
\quad \text{(cell state update)}
$$

$$
\mathbf{o}_t =
\sigma(\mathbf{W}_o\mathbf{x}_t +
\mathbf{U}_o\mathbf{h}_{t-1} + \mathbf{b}_o)
\quad \text{(output gate)}
$$

$$
\mathbf{h}_t =
\mathbf{o}_t \odot \tanh(\mathbf{C}_t)
\quad \text{(hidden state update)}
$$

Here, \(\sigma\) is the sigmoid function and \(\odot\) is element-wise
multiplication. The final hidden representation is passed to a linear dense
output layer to produce the next-day price estimate.

The training objective is mean squared error with L2 regularization:

$$
\mathcal{L}(\theta) =
\frac{1}{N}\sum_{k=1}^{N}(y_k-\hat{y}_k)^2
+ \lambda\sum_{\ell}\|\mathbf{W}_\ell\|_2^2
$$

where \(\lambda=10^{-3}\) and the second term penalizes excessively large
weights.

## 3. Architecture and Training Configuration

```text
[Input: 10 timesteps x 1 price feature]
                     |
                     v
[LSTM: 32 units, return_sequences=True, L2=0.001]
                     |
                     v
[Dropout: 30%]
                     |
                     v
[LSTM: 16 units, return_sequences=False, L2=0.001]
                     |
                     v
[Dropout: 20%]
                     |
                     v
[Dense: 1 unit, linear activation]
                     |
                     v
[Next-day closing-price prediction]
```

| Configuration | Value |
|---|---|
| Input feature | NDX `Close` price only |
| Lookback window | 10 trading days |
| Split | 60% train, 20% validation, 20% test |
| LSTM layers | 32 units, then 16 units |
| Dropout | 30% after the first LSTM, 20% after the second |
| Kernel regularization | L2 = 0.001 |
| Optimizer | Adam, learning rate 0.001 |
| Loss | Mean squared error |
| Early stopping | Validation loss, patience 25 |
| Learning-rate reduction | Factor 0.5, patience 10 |
| Random seed | NumPy and TensorFlow seed 42 |

## 4. Leakage Prevention and Evaluation

The data is split chronologically rather than randomly. The `MinMaxScaler` is
fit only on the training prices; validation and test prices are transformed
using those fixed training parameters. This prevents future price information
from influencing the preprocessing step.

The backtest uses the prediction available at time \(t\) to compare against the
actual next price \(p_{t+1}\). A long position is taken when the predicted next
price is above the current price; otherwise the strategy takes a short
direction. The shared implementation is in
[`src/backtest_engine.py`](../src/backtest_engine.py).

Metrics are saved in [`results/lstm_metrics.json`](../results/lstm_metrics.json)
and include:

- MAE, RMSE, and MAPE for price prediction error
- Directional accuracy for up/down movement
- Strategy return and Sharpe ratio
- Buy-and-hold return for market comparison

The generated Standard LSTM artifacts are
[`models/lstm_model.keras`](../models/lstm_model.keras) and
[`results/lstm_model_results.png`](../results/lstm_model_results.png).

## 5. Role as a Baseline and Limitations

This model deliberately uses only one price feature and a plain stacked LSTM.
It therefore provides a controlled baseline: improvements from sentiment or
attention models can be evaluated against the same chronological split and
sequence setup instead of against a more complex architecture.

The model does not observe news, macroeconomic variables, technical indicators,
or order-book information. It also predicts a noisy continuous price rather
than directly optimizing trading profit. Its results should therefore be
interpreted as an out-of-sample baseline, not as evidence of a guaranteed
trading strategy.

