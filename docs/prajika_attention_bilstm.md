# Model 4: Attention-BiLSTM Architecture & Theoretical Formulation

**Course:** Deep Learning (ICT-4442) — Academic Year 2026–2027  
**Department:** School of Computer Engineering, Manipal Institute of Technology  
**Author:** Prajika Saini (Registration No. 230953106)  
**Git Branch:** `feat/prajika-attention-bilstm`  
**Assigned Deliverable:** Model 4 — Multi-Modal Bidirectional LSTM with Additive Temporal Attention & Residual Skip Connections  

---

## 1. Executive Summary

Financial asset price series are characterized by non-stationary dynamics, high noise-to-signal ratios, and sudden volatility regime shifts driven by unscheduled news announcements. Standard recurrent neural networks and unidirectional LSTMs suffer from temporal attenuation over extended lookback sequences, often failing to isolate critical shock events from routine market fluctuations.

This module implements **Model 4: Attention-BiLSTM**, an advanced sequence-processing architecture authored by **Prajika Saini (230953106)**. The network ingests 10-day lookback sequences combining normalized closing prices of the NASDAQ-100 index (`^NDX`) and daily sentiment scores derived from financial news headlines via ProsusAI/FinBERT. The architecture features:
1. A **Bidirectional LSTM (BiLSTM)** layer that simultaneously aggregates forward historical momentum and backward contextual dependencies.
2. A **linear residual skip projection** that preserves gradient flow and retains raw feature magnitude across recurrent transformations.
3. A **feedforward Temporal Attention Mechanism** that dynamically assigns non-uniform importance weights ($\alpha_t$) across the 10 trading timesteps.
4. A dense regression head predicting next-day close, evaluated under an identical chronological 60/20/20 train-validation-test split and an automated trading backtest.

---

## 2. Mathematical Formulation & Architecture

### 2.1 Input Representation
Let $\mathbf{x}_t \in \mathbb{R}^D$ denote the feature vector at trading day $t$, where $D=2$ (normalized Close price and daily FinBERT sentiment score). For each prediction step, the input sequence spans a 10-day historical window:

$$\mathbf{X} = [\mathbf{x}_{t-9}, \mathbf{x}_{t-8}, \dots, \mathbf{x}_t]^\top \in \mathbb{R}^{T \times D}, \quad T = 10$$

### 2.2 Bidirectional LSTM (BiLSTM)
A standard LSTM maintains an input gate $\mathbf{i}_t$, forget gate $\mathbf{f}_t$, output gate $\mathbf{o}_t$, and cell state $\mathbf{c}_t$. To capture past momentum and bidirectional contextual correlations, two independent recurrent passes are executed:

- **Forward LSTM:** Processes the sequence from $k=1 \dots T$:

$$\overrightarrow{\mathbf{h}}_t = \text{LSTM}_{\text{fwd}}(\mathbf{x}_t, \overrightarrow{\mathbf{h}}_{t-1}), \quad \overrightarrow{\mathbf{h}}_t \in \mathbb{R}^{32}$$

- **Backward LSTM:** Processes the sequence in reverse chronological order from $k=T \dots 1$:

$$\overleftarrow{\mathbf{h}}_t = \text{LSTM}_{\text{bwd}}(\mathbf{x}_t, \overleftarrow{\mathbf{h}}_{t+1}), \quad \overleftarrow{\mathbf{h}}_t \in \mathbb{R}^{32}$$

The hidden states are concatenated at each timestep:

$$\mathbf{H}_t = [\overrightarrow{\mathbf{h}}_t \mathbin{\Vert} \overleftarrow{\mathbf{h}}_t] \in \mathbb{R}^{64}, \quad \mathbf{H} \in \mathbb{R}^{T \times 64}$$

### 2.3 Residual Skip Projection
To safeguard against gradient vanishing during backpropagation through time and prevent information degradation of raw signals, a linear projection aligns the input dimension $D=2$ to the recurrent hidden dimension $2H=64$:

$$\mathbf{H}_{\text{proj}} = \mathbf{X} \mathbf{W}_{\text{res}} + \mathbf{b}_{\text{res}}, \quad \mathbf{W}_{\text{res}} \in \mathbb{R}^{D \times 64}, \;\; \mathbf{b}_{\text{res}} \in \mathbb{R}^{64}$$

$$\mathbf{Z} = \mathbf{H} + \mathbf{H}_{\text{proj}} \in \mathbb{R}^{T \times 64}$$

### 2.4 Temporal Additive Attention Layer
Rather than retaining only the final recurrent state $\mathbf{Z}_T$, an additive attention mechanism evaluates the relative salience of each timestep $\tau \in \{1, \dots, T\}$. A learned projection vector $\mathbf{v}$ and weight matrix $\mathbf{W}_a$ score each hidden state:

$$e_\tau = \mathbf{v}^\top \tanh(\mathbf{W}_a \mathbf{z}_\tau + \mathbf{b}_a), \quad \mathbf{W}_a \in \mathbb{R}^{64 \times 64}, \;\; \mathbf{v} \in \mathbb{R}^{64}$$

The alignment scores are normalized across the time horizon via the Softmax operator:

$$\alpha_\tau = \frac{\exp(e_\tau)}{\sum_{k=1}^T \exp(e_k)}, \quad \sum_{\tau=1}^T \alpha_\tau = 1, \;\; \alpha_\tau \ge 0$$

The final context vector $\mathbf{c} \in \mathbb{R}^{64}$ is computed as the probability-weighted sum:

$$\mathbf{c} = \sum_{\tau=1}^T \alpha_\tau \mathbf{z}_\tau$$

### 2.5 Regression Head
The context vector $\mathbf{c}$ passes through a non-linear dense projection with Dropout regularization before generating the scalar prediction $\hat{y}_{t+1}$:

$$\mathbf{h}_{\text{dense}} = \text{ReLU}(\mathbf{W}_d \mathbf{c} + \mathbf{b}_d), \quad \mathbf{W}_d \in \mathbb{R}^{64 \times 32}$$

$$\mathbf{h}_{\text{drop}} = \text{Dropout}(\mathbf{h}_{\text{dense}}, p=0.2)$$

$$\hat{y}_{t+1} = \mathbf{w}_o^\top \mathbf{h}_{\text{drop}} + b_o, \quad \mathbf{w}_o \in \mathbb{R}^{32}$$

---

## 3. Evaluation & Backtest Protocol

### 3.1 Loss Function & Chronological Splitting
The network is optimized via the Adam optimizer ($\eta = 0.001$) under Mean Squared Error (MSE) loss:

$$\mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$$

Data is split chronologically into 60% Train, 20% Validation, and 20% Out-of-Sample Test. MinMax scalers are fitted exclusively on the training partition to prevent lookahead data leakage.

### 3.2 Trading Simulation & Sharpe Ratio
Trading signals are generated according to predicted directional sign:

$$\text{Signal}_t = \begin{cases} 1 & \text{if } \hat{y}_{t+1} > y_t \\ 0 & \text{otherwise} \end{cases}$$

Strategy returns:

$$R_t^{\text{strategy}} = \text{Signal}_t \times \left( \frac{y_{t+1} - y_t}{y_t} \right)$$

Annualized Sharpe Ratio ($\text{Sharpe}$), assuming a risk-free rate of zero:

$$\text{Sharpe} = \frac{\mathbb{E}[R^{\text{strategy}}]}{\sigma(R^{\text{strategy}})} \times \sqrt{252}$$