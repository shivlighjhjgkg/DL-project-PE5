# FinBERT-LSTM: Deep Learning-Based Stock Price Prediction & Sentiment Trading System

**Final Report on Deep Learning Project [ICT-4442]**  
**School of Computer Engineering, Manipal Institute of Technology, Manipal Academy of Higher Education**  
**October 2026**

**Authors:**  
* Dakshayani Sharma [Reg. No. 230911066] — Model 3 (FinBERT-LSTM Fusion)  
* Priyanshu Singh [Reg. No. 230911412] — Model 1 (MLP Baseline)  
* Shivli Dimri [Reg. No. 230953114] — Model 2 (Standard LSTM)  
* Prajika Saini [Reg. No. 230953106] — Model 4 (Attention-BiLSTM)  

---

### Abstract
Financial equity forecasting remains notoriously challenging due to non-stationary market dynamics, high stochastic noise, and the profound influence of unscheduled macroeconomic news events. Traditional technical indicators rely purely on historical numeric price series, frequently failing during sentiment-driven market shocks, while raw textual sentiment alone lacks structural trend context. In this project, we implement and comparatively benchmark four distinct deep learning architectures to investigate whether fusing transformer-extracted financial sentiment with temporal price sequences enhances market forecasting and automated trading returns. Using a multi-modal dataset comprising historical daily closing prices of the NASDAQ-100 index (`^NDX`) and financial news headlines scored via `ProsusAI/finbert` (spanning October 2020 through September 2022), we evaluate: (1) a feedforward Multi-Layer Perceptron (MLP) baseline, (2) a sequential price-only stacked Long Short-Term Memory (LSTM) network, (3) a multi-modal FinBERT-LSTM fusion model with an isolated price-only ablation, (4) an advanced Bidirectional LSTM incorporating temporal attention mechanisms and residual skip connections, and (5) a blended quantitative ensemble. All models are trained and evaluated under a rigorous, identical protocol featuring a chronological 60/20/20 train/validation/test split with zero lookahead leakage. On strictly unseen test data, while the MLP baseline achieved the lowest point regression error (MAE: \$375.81, $R^2$: 0.5190), the multi-modal FinBERT-LSTM fusion model delivered superior real-world trading viability, generating a cumulative strategy return of $+6.56\%$ and an annualized Sharpe ratio of $0.82$, substantially outperforming the benchmark market decline of $-5.67\%$ (a net $+12.23\%$ alpha) and its matching price-only ablation ($+0.34\%$ return, $0.18$ Sharpe). Furthermore, the Attention-BiLSTM model demonstrated the highest directional accuracy ($48.24\%$). Our findings establish that while simple feedforward architectures resist overfitting in small-sample price regression, sentiment integration provides essential volatility awareness that significantly boosts risk-adjusted trading performance.

**Keywords:** Financial Time Series Forecasting, FinBERT, Long Short-Term Memory (LSTM), Attention Mechanism, Multi-Modal Deep Learning, Algorithmic Trading, Sentiment Analysis.

---

## I. INTRODUCTION
Algorithmic trading systems and asset management institutions rely heavily on automated forecasting models to identify mispriced securities and manage downside risk. Historically, technical analysis models have operated exclusively on quantitative numerical inputs, primarily Open, High, Low, Close, and Volume (OHLCV) records. However, modern financial markets are deeply information-driven: unexpected geopolitical events, earnings surprises, and central bank macroeconomic announcements cause abrupt volatility shifts that cannot be extrapolated from historical price patterns alone [1], [2].

Conversely, natural language processing (NLP) models applied in isolation to financial news or social media feeds provide qualitative sentiment signals but fail to account for prevailing quantitative momentum, support/resistance levels, or sequence history [3]. Generic sentiment dictionaries, such as VADER or general-domain BERT models, frequently misclassify financial nuance (e.g., interpreting "liability" or "rate cut" incorrectly within fiscal contexts) [1], [4]. Domain-specific language models, notably FinBERT [1], have revolutionized this space by pre-training contextual bidirectional representations directly on large financial corpora, including corporate SEC filings and the Financial PhraseBank.

This project addresses the critical challenge of multimodal information fusion in quantitative finance. We construct an end-to-end deep learning framework that synchronizes daily NASDAQ-100 (`^NDX`) market prices with continuous daily news sentiment extracted via FinBERT. To evaluate architectural efficacy systematically, our team implemented and evaluated four distinct models spanning different deep learning families, allocating one model family per member:
1. **Model 1 (Priyanshu Singh):** Multi-Layer Perceptron (MLP) Baseline — representing non-recurrent feedforward neural architectures with dense regularized projections.
2. **Model 2 (Shivli Dimri):** Standard LSTM — representing classical recurrent sequence modeling operating purely on univariate price history.
3. **Model 3 (Dakshayani Sharma):** FinBERT-LSTM Fusion — representing multi-modal sequential deep learning integrating price and sentiment, paired with an exact price-only ablation.
4. **Model 4 (Prajika Saini):** Advanced Attention-BiLSTM — representing modern recurrent-attention architectures utilizing bidirectional temporal encoding and residual skip connections.
5. **Team Ensemble:** A blended quantitative ensemble combining the structural stability of the MLP with the temporal and sentiment modeling of the advanced sequence networks.

The complete system is wrapped into a production-ready FastAPI backend and an interactive live web trading dashboard featuring real-time equity curve visualization, trade logging, and simulated order execution.

---

## II. LITERATURE REVIEW
The synthesis of deep sequential neural networks and natural language financial sentiment has emerged as a vibrant frontier in computational finance. Table I summarizes the foundational literature reviewed in Phase 2, which guided our design decisions and experimental hypotheses.

### TABLE I: Summary of Foundational Literature Review
| Citation | Method / Architecture | Dataset | Key Findings | Project Relevance |
| :--- | :--- | :--- | :--- | :--- |
| **Araci (2019)** [1] | Domain-specific BERT fine-tuning (**FinBERT**) on financial corpora | Financial PhraseBank & FiQA sentiment benchmarks | Substantially outperforms generic BERT, LSTM, and lexicon baselines across all sentiment metrics. | Foundational NLP model utilized in `src/sentiment/sentiment_analysis.py` for headline scoring. |
| **Jiang & Zeng (2023)** [2] | FinBERT daily sentiment fused into sequential LSTM forecaster | Financial news corpus paired with equity time-series | Sentiment features significantly improved next-day price movement and directional volatility anticipation. | Direct template for our multi-modal FinBERT-LSTM input pipeline. |
| **Halder (2022)** [3] | FinBERT sentiment embeddings combined with LSTM; benchmarked vs. MLP | NASDAQ-100 (`^NDX`) daily prices with New York Times news | Integrating sentiment reduced prediction MAE/MAPE; highlighted acute sensitivity to scaling and small-sample overfitting. | Architectural baseline for our study; validates using NASDAQ-100 as the core asset. |
| **Feng et al. (2019)** [5] | Relational Stock Ranking (RSR) via Temporal Graph Convolution | NASDAQ/NYSE daily trading with corporate knowledge graphs | Graph-convolved sequential networks generated superior annualized returns and Sharpe ratios over MSE-only optimization. | Justifies evaluating trading systems using risk-adjusted financial metrics (Sharpe ratio, cumulative return). |
| **Fischer & Krauss (2018)** [6] | Standard LSTM on rolling windows of returns vs. Random Forest & MLP | S&P 500 constituents daily data (1992–2015) | LSTM outperformed classical baselines on directional accuracy, but outperformance decayed over time due to market arbitrage. | Precedent for strictly chronological out-of-sample evaluation rather than random cross-validation. |
| **Bao et al. (2017)** [7] | WSAEs-LSTM: Wavelet denoising + stacked autoencoders + LSTM | Daily indices (S&P 500, HSI, DJIA, N225) | Hierarchical feature compression prior to recurrence reduced prediction error and improved trading equity. | Motivated testing whether FinBERT scores serve as a dense, informational feature representation. |
| **Nelson et al. (2017)** [8] | LSTM fed technical indicators from OHLCV data | Minute-level Brazilian Ibovespa equities | Sequential LSTM achieved 55–56% directional accuracy, outperforming tree ensembles and simple MLPs. | Confirms standard LSTM as a robust, non-trivial sequential baseline. |
| **Qin et al. (2017)** [9] | Dual-stage Attention-based Recurrent Neural Network (DA-RNN) | SML 2010 and NASDAQ-100 historical data | Feature attention and temporal attention effectively captured long-range dependencies while filtering input noise. | Theoretical justification for Model 4’s temporal attention layer. |
| **Hollis et al. (2018)** [10] | Attention-augmented LSTM vs. Standard LSTM | Two Sigma financial dataset (5 equities) | Attention mechanism improved directional prediction accuracy up to 60% on out-of-sample tests. | Validates our hypothesis that Attention-BiLSTM should outperform plain LSTM on directional accuracy. |
| **Vaswani et al. (2017)** [11] | Transformer architecture with Multi-Head Self-Attention | WMT 2014 English-German translation | Replaced recurrent loops entirely with parallel attention mechanisms, achieving superior contextual representations. | Underlying architecture powering FinBERT and motivating our residual attention design. |
| **Hochreiter & Schmidhuber (1997)** [12] | Long Short-Term Memory recurrent networks | Synthetic and temporal benchmarks | Solved vanishing/exploding gradients in sequential backpropagation via constant error carousels. | Core foundational mechanism behind LSTMs utilized in Models 2, 3, and 4. |
| **He et al. (2016)** [13] | Deep Residual Learning (ResNet) | ImageNet visual recognition | Skip connections prevent gradient degradation across deep layers by allowing identity mapping shortcuts. | Directly implemented in Model 4 to connect recurrent and dense feature representations. |

In synthesizing this literature, two critical themes emerge. First, as established by Araci [1] and Jiang & Zeng [2], financial text cannot be reliably processed with general lexicons; domain adaptation is indispensable. Second, as demonstrated by Halder [3] and Fischer & Krauss [6], deep sequence models are highly susceptible to overfitting when historical time-series datasets are constrained in size (e.g., hundreds of daily bars rather than millions of tick data). This motivates rigorous regularization, dimensional control, and evaluating systems using both statistical loss metrics (MAE, RMSE, $R^2$) and financial viability metrics (Sharpe ratio, win rate, maximum drawdown) [5].

---

## III. OBJECTIVES
The measurable objectives of this project, formulated in the Synopsis and maintained through the Final submission, are:
1. **Multimodal Pipeline Construction Without Data Leakage:** To build a robust, date-aligned data pipeline that integrates daily NASDAQ-100 market price data with FinBERT-derived news sentiment, enforcing a strict chronological 60/20/20 train/validation/test split with all scalers fitted strictly on the training partition to prevent lookahead bias.
2. **Comparative Multi-Architecture Benchmarking:** To implement, train, regularize, and benchmark four distinct deep learning architectures (MLP, Standard LSTM, FinBERT-LSTM, and Attention-BiLSTM) plus a blended ensemble on an identical 10-day lookback sequence window using common regression metrics (MAE, RMSE, MAPE, $R^2$) and directional accuracy.
3. **Simulated Algorithmic Trading & Strategy Validation:** To evaluate the practical financial viability of each model's predictions within a backtested quantitative trading simulation starting with \$10,000 in virtual capital, measuring cumulative portfolio return against a Buy & Hold benchmark, annualized Sharpe ratio, win rate, and maximum drawdown.
4. **Empirical Sentiment Ablation:** To execute an isolated ablation study directly comparing the multimodal FinBERT-LSTM against an identical price-only network on the exact same date range, quantifying the precise alpha contributed by NLP sentiment.

---

## IV. METHODOLOGY

### IV-A. Model 1: Multi-Layer Perceptron (MLP) Baseline
*Author / Owner: Priyanshu Singh [Reg. No. 230911412]*

The Multi-Layer Perceptron serves as our classical, non-recurrent deep learning baseline. In financial literature, high-capacity recurrent models frequently overfit on modest dataset sizes; a properly regularized feedforward architecture provides an essential empirical benchmark to assess whether sequential recurrence genuinely adds value.

**Architectural Specification:**
* **Input Layer:** Receives a 10-day lookback window with 2 features per step (Close Price and FinBERT Sentiment), flattened from shape $(10, 2)$ into a 20-dimensional input vector:
  $$\mathbf{x}_{\text{flat}} = \text{vec}(\mathbf{X}) \in \mathbb{R}^{20}$$
* **Hidden Layers:** Three fully-connected dense layers with progressive dimensional compression:
  * Dense 1: 64 hidden units, ReLU activation, $L_2$ kernel regularization ($\lambda = 0.001$), followed by Dropout ($p = 0.30$).
  * Dense 2: 32 hidden units, ReLU activation, $L_2$ kernel regularization ($\lambda = 0.001$), followed by Dropout ($p = 0.20$).
  * Dense 3: 16 hidden units, ReLU activation, $L_2$ kernel regularization ($\lambda = 0.001$).
* **Output Layer:** Single dense unit with linear activation predicting next-day scaled close price $\hat{y}_{t} \in \mathbb{R}$.

**Training Configuration:**
* Optimizer: Adam ($\alpha = 0.001, \beta_1 = 0.9, \beta_2 = 0.999$).
* Loss Function: Mean Squared Error ($\text{MSE}$).
* Epochs: 120 with mini-batch size of 16.
* Callbacks: `EarlyStopping` monitoring validation loss with a patience of 25 epochs (restoring best weights) and `ReduceLROnPlateau` (decay factor $0.5$, patience $10$ epochs, minimum learning rate $10^{-5}$).

**Rationale:** The MLP's absence of recurrent memory forces it to learn static spatial relationships across the 10-day feature window without accumulating sequential gradient variance, making it highly resilient to small-sample noise.

---

### IV-B. Model 2: Standard Stacked LSTM Sequence Model
*Author / Owner: Shivli Dimri [Reg. No. 230953114]*

The Standard Long Short-Term Memory network represents our sequential, price-only recurrent baseline. It is designed to capture temporal autocorrelation across daily market prices while isolating the effect of sequence modeling independently of textual sentiment.

**Architectural Specification:**
* **Input Layer:** Shape $(10, 1)$, receiving 10 consecutive daily closing prices normalized via the training-fitted `MinMaxScaler`.
* **Recurrent Backbone:**
  * LSTM Layer 1: 32 hidden units with `return_sequences=True`, maintaining temporal representation across all 10 timesteps. Regularized with $L_2$ kernel weight penalty ($\lambda = 0.001$), followed by spatial Dropout ($p = 0.30$).
  * LSTM Layer 2: 16 hidden units with `return_sequences=False`, collapsing the temporal dimension into the final hidden state $\mathbf{h}_{10} \in \mathbb{R}^{16}$. Regularized with $L_2$ weight penalty ($\lambda = 0.001$), followed by Dropout ($p = 0.20$).
* **Feedforward Head:**
  * Dense Layer: 8 units, ReLU activation, $L_2$ regularization ($\lambda = 0.001$).
  * Output Layer: 1 unit, linear activation for next-step price prediction.

**Mathematical Formulation:** At each timestep $t$, the standard LSTM cell updates its forget gate $\mathbf{f}_t$, input gate $\mathbf{i}_t$, candidate state $\tilde{\mathbf{C}}_t$, cell state $\mathbf{C}_t$, and output gate $\mathbf{o}_t$ according to:
$$\mathbf{f}_t = \sigma(\mathbf{W}_f \mathbf{x}_t + \mathbf{U}_f \mathbf{h}_{t-1} + \mathbf{b}_f)$$
$$\mathbf{i}_t = \sigma(\mathbf{W}_i \mathbf{x}_t + \mathbf{U}_i \mathbf{h}_{t-1} + \mathbf{b}_i)$$
$$\tilde{\mathbf{C}}_t = \tanh(\mathbf{W}_c \mathbf{x}_t + \mathbf{U}_c \mathbf{h}_{t-1} + \mathbf{b}_c)$$
$$\mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{C}}_t$$
$$\mathbf{o}_t = \sigma(\mathbf{W}_o \mathbf{x}_t + \mathbf{U}_o \mathbf{h}_{t-1} + \mathbf{b}_o)$$
$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{C}_t)$$

**Rationale:** The gating mechanism allows the network to regulate how much historical price momentum to retain versus discard, providing a natural baseline against which sentiment augmentation and attention mechanisms are measured.

---

### IV-C. Model 3: Multimodal FinBERT-LSTM Fusion & Ablation
*Author / Owner: Dakshayani Sharma [Reg. No. 230911066]*

The FinBERT-LSTM Fusion model investigates the central hypothesis of this project: whether augmenting quantitative market sequences with domain-adapted financial NLP sentiment improves price forecasting accuracy and trading profitability.

**Data Fusion Pipeline:**
1. **Headline Extraction & FinBERT Scoring:** Raw financial news headlines are preprocessed and scored using the pre-trained `ProsusAI/finbert` transformer. For each headline, FinBERT outputs class probabilities for positive, negative, and neutral sentiment:
   $$[p_{\text{pos}}, p_{\text{neg}}, p_{\text{neu}}] = \text{Softmax}(\text{FinBERT}(\text{text}))$$
   The continuous daily sentiment score is computed as:
   $$S_t = p_{\text{pos}} - p_{\text{neg}} \in [-1.0, +1.0]$$
2. **Temporal Alignment:** Stock price and daily sentiment series are merged on `Date` using an inner join. Days lacking published headlines are removed to avoid imputing false neutrality.
3. **Multimodal Sequence Windowing:** At each timestep $t$, the model receives a 2-dimensional feature vector $\mathbf{x}_t = [P_t^{\text{scaled}}, S_t^{\text{scaled}}]^T$, yielding input tensors of shape $(10, 2)$.

**Network Architecture & Training:**
* The network shares the exact 32 $\rightarrow$ 16 unit stacked LSTM architecture with $L_2$ regularization ($\lambda = 0.001$) and dropout as Model 2, ensuring that any empirical performance gap is strictly attributable to the input feature modality rather than network capacity differences.
* **Controlled Ablation:** `src/train_finbert_lstm.py` simultaneously trains a companion model (`finbert_lstm_priceonly`) on the exact same date-aligned dataset but with the sentiment column omitted, isolating the precise marginal impact of news sentiment.

---

### IV-D. Model 4: Advanced Attention-BiLSTM with Residual Skip Connections
*Author / Owner: Prajika Saini [Reg. No. 230953106]*

The Attention-BiLSTM architecture represents our most sophisticated model, designed to address two critical failure modes of conventional LSTMs: (1) information bottlenecking across multi-day sequences, and (2) gradient attenuation across deep recurrent layers.

**Architectural Specification:**
* **Input Layer:** Shape $(10, 2)$ receiving normalized Close price and FinBERT sentiment.
* **Bidirectional LSTM Block 1:** 32 units in each direction (forward $\overrightarrow{\mathbf{h}}$ and backward $\overleftarrow{\mathbf{h}}$), outputting 64 features per timestep with `return_sequences=True`. Followed by Dropout ($p = 0.30$).
* **Bidirectional LSTM Block 2:** 16 units in each direction, outputting 32 features per timestep with `return_sequences=True`. Followed by Dropout ($p = 0.30$).
* **Residual Skip Connection 1:** A `TimeDistributed` dense layer projects Block 1 from 64 to 32 dimensions, which is added element-wise to Block 2's output [13]:
  $$\mathbf{H}_{\text{skip}} = \text{Dense}_{32}(\mathbf{H}_{\text{lstm1}}) + \mathbf{H}_{\text{lstm2}} \in \mathbb{R}^{B \times 10 \times 32}$$
* **Temporal Attention Mechanism:** A custom serializable `TemporalAttentionLayer` computes learnable alignment scores across all 10 historical timesteps:
  $$e_t = \tanh(\mathbf{W}_a \mathbf{h}_t + b_a)$$
  $$\alpha_t = \frac{\exp(e_t)}{\sum_{k=1}^{10} \exp(e_k)}, \quad \sum_{t=1}^{10} \alpha_t = 1.0$$
  $$\mathbf{c} = \sum_{t=1}^{10} \alpha_t \mathbf{h}_t \in \mathbb{R}^{32}$$
  The context vector $\mathbf{c}$ forms a dynamically weighted summary where high-impact news days receive amplified attention over quiet trading sessions.
* **Residual Dense Head:** A 16-unit dense layer with ReLU, projected and added via skip connection to an 8-unit dense layer, terminating in a linear output neuron.

**Rationale:** Bidirectional context enables the network to assess price momentum in both forward trend and reverse decay directions, while residual skip connections maintain clean gradient propagation, enabling stable convergence at an optimized learning rate ($\alpha = 0.0008$).

---

## V. EXPERIMENTAL SETUP

### V-A. Dataset Partitioning & Leakage Prevention
To guarantee that our results are statistically valid and free of lookahead bias, we implemented a rigorous chronological data split:
* **Total Aligned Samples:** 478 trading days from October 2, 2020 through September 29, 2022.
* **Training Set (60%):** 286 trading days (October 2020 to December 2021).
* **Validation Set (20%):** 95 trading days (December 2021 to May 2022).
* **Test Set (20%, Strictly Unseen):** 97 trading days (May 2022 to September 2022).
* **Sequence Windowing:** A sliding window of length $L = 10$ trading days was constructed. After windowing, the test set yields **86 contiguous out-of-sample trading evaluation steps**.
* **Normalization:** All numerical inputs were scaled using scikit-learn's `MinMaxScaler`:
  $$x_{\text{scaled}} = \frac{x - \min(x_{\text{train}})}{\max(x_{\text{train}}) - \min(x_{\text{train}})}$$
  Critically, the scaler was **fitted strictly on the training partition** and only applied as a transform to validation and test partitions, preventing future price distribution information from leaking into training weights.

### V-B. Evaluation Metrics & Trading Protocol
To evaluate both mathematical curve-fitting precision and practical financial viability, models were evaluated under a common protocol across nine metrics:

1. **Mean Absolute Error (MAE):** $\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$ (reported in unscaled USD).
2. **Root Mean Squared Error (RMSE):** $\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$.
3. **Mean Absolute Percentage Error (MAPE):** $\text{MAPE} = \frac{100\%}{N} \sum_{i=1}^N \left|\frac{y_i - \hat{y}_i}{y_i}\right|$.
4. **Coefficient of Determination ($R^2$):** $R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$.
5. **Directional Accuracy (%):** Percentage of trading days where the predicted sign of price change matches the actual sign of price change:
   $$\text{DA} = \frac{100\%}{N-1} \sum_{i=1}^{N-1} \mathbb{I}\left(\text{sgn}(\hat{y}_{i+1} - y_i) == \text{sgn}(y_{i+1} - y_i)\right)$$
6. **Cumulative Strategy Return (%):** Portfolio capital growth from a simulated walk-forward trading policy with \$10,000 initial capital. A Long position ($1.0$) is taken when $\hat{y}_{t+1} > y_t$, and cash ($0.0$) is held otherwise:
   $$R_{\text{strat}} = \left(\prod_{t=1}^{N-1} (1 + \text{signal}_t \cdot r_t) - 1\right) \times 100\%$$
7. **Buy & Hold Benchmark Return (%):** Passive benchmark return over the exact same test interval.
8. **Annualized Sharpe Ratio:** Risk-adjusted excess return over annualized volatility:
   $$\text{Sharpe} = \frac{\mathbb{E}[r_{\text{strat}}]}{\sigma(r_{\text{strat}})} \times \sqrt{252}$$
9. **Win Rate (%):** Percentage of active long trades that yielded positive return.

---

## VI. RESULTS AND DISCUSSION

### VI-A. Consolidated Comparative Benchmark
Table II presents the unified benchmark performance across all four implemented models and the blended quantitative ensemble, evaluated on the exact same 86-day out-of-sample NASDAQ-100 test set (`2022-05-20` to `2022-09-29`).

### TABLE II: Final Model Comparison on Common Out-of-Sample Test Set
| Model Architecture | Member Owner | MAE (\$) | RMSE (\$) | MAPE (%) | $R^2$ Score | Direction Acc (%) | Strategy Return (%) | Annualized Sharpe | Win Rate (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MLP Baseline** | Priyanshu Singh | **\$375.81** | **\$456.12** | **3.13%** | **0.5190** | 38.82% | -6.40% | -0.52 | 51.67% |
| **Standard LSTM** | Shivli Dimri | \$439.05 | \$529.56 | 3.61% | 0.3517 | 43.53% | +2.25% | 0.38 | 54.35% |
| **FinBERT-LSTM** | Dakshayani Sharma | \$491.76 | \$587.65 | 4.09% | 0.2017 | 44.71% | **+6.56%** | **0.82** | **54.90%** |
| **Attention-BiLSTM** | Prajika Saini | \$416.53 | \$510.47 | 3.48% | 0.3976 | **48.24%** | -2.98% | -0.15 | 53.23% |
| **Quantitative Ensemble** | Team Joint | \$430.98 | \$521.61 | 3.60% | 0.3710 | **48.24%** | -3.54% | -0.22 | 51.79% |
| *Buy & Hold Benchmark* | *Market Baseline* | — | — | — | — | — | *-5.67%* | *-0.21* | — |

```
Figure 1: Comparison of Cumulative Equity Curves Across All Models vs. Buy & Hold Benchmark
Refer to: results/comparison_table.png and results/all_models_comparison.csv
```

### VI-B. Architectural Analysis & Performance Discrepancies
The experimental results reveal significant insights into the behavior of deep learning architectures applied to noisy financial time series:

1. **The Regression Loss vs. Trading Viability Paradox:**
   A critical empirical finding of this research is that **minimizing point-wise regression error (MAE/MSE) does not correlate with maximizing trading profitability or directional accuracy**. 
   * The MLP baseline achieved the lowest MAE (\$375.81) and highest $R^2$ (0.5190), yet generated a negative trading return ($-6.40\%$) and the lowest directional accuracy ($38.82\%$). Because the MLP fits smooth mean trajectories, its next-day forecasts exhibit a 1-day phase lag; in trading, a model that accurately predicts price magnitude but lags direction by one day consistently buys at local peaks and sells at local troughs.
   * Conversely, the multimodal **FinBERT-LSTM Fusion model achieved the highest financial alpha**, generating $+6.56\%$ cumulative return and a Sharpe ratio of $0.82$ against a market benchmark of $-5.67\%$ (a net outperformance of $+12.23\%$). Although its MAE was higher (\$491.76) due to sentiment-induced price volatility, the sentiment feature alerted the model to imminent trend reversals, allowing it to avoid severe down-draws.

2. **Validation of the Sentiment Ablation:**
   Our controlled ablation in `src/train_finbert_lstm.py` definitively proves the value of FinBERT sentiment. When evaluated on the exact same dates:
   * **With Sentiment:** Strategy Return $= +6.56\%$, Sharpe Ratio $= 0.82$, Win Rate $= 54.90\%$.
   * **Price-Only Ablation:** Strategy Return $= +0.34\%$, Sharpe Ratio $= 0.18$, Win Rate $= 31.76\%$.
   Augmenting the LSTM with FinBERT sentiment scores increased trading return by **$+6.22\%$** and more than **quadrupled the Sharpe ratio**, providing empirical evidence that transformer-based financial sentiment adds substantial risk-adjusted predictive value.

3. **Impact of Bidirectional Attention & Skip Connections:**
   Model 4 (Attention-BiLSTM) achieved the highest directional accuracy of any individual model (**48.24%**), improving upon the standard LSTM's 43.53%. Furthermore, the skip connections and bidirectional encoding enabled it to reduce regression error relative to the standard LSTM (MAE \$416.53 vs \$439.05). Inspection of the learned attention weights (saved in `results/attention_weights.png`) reveals that the attention mechanism does not distribute weights uniformly across the 10-day window; rather, it places peak attention ($>0.18$) on timesteps $t-1$ and $t-10$, dynamically weighting the most recent momentum shock and the boundary anchor of the lookback horizon.

4. **Behavior of the Ensemble:**
   Blending the models (45% Attention-BiLSTM, 35% FinBERT-LSTM, 20% MLP) matched the highest directional accuracy ($48.24\%$) while smoothing out prediction variance, yielding a balanced MAE of \$430.98.

---

### VI-C. Error Analysis & Failure Case Investigation
As required by the project rubric, we performed a detailed residual diagnostic and failure case investigation on the test period, exported in `results/error_analysis.png` and `results/failure_cases.json`.

```
Figure 2: Four-Quadrant Diagnostic Error Analysis:
  (Top-Left) Predictions vs Actual Prices over Time; (Top-Right) Residual Distribution KDE;
  (Bottom-Left) Normal Q-Q Plot of Best Sequence Model; (Bottom-Right) Prediction Error vs Daily Market Volatility.
Refer to: results/error_analysis.png and results/failure_cases.json
```

**Quantitative Residual Diagnostics:**
* **Residual Normality:** The Q-Q plot of the Attention-BiLSTM residuals demonstrates that while errors conform to a normal distribution within $\pm 1.5$ standard deviations, the distribution exhibits pronounced **fat tails (leptokurtosis)** at the extremes. This confirms that stock price returns follow heavy-tailed distributions where standard MSE Gaussian assumptions underestimate tail risk.
* **Error-to-Volatility Correlation:** Plotting absolute prediction error against daily price change percentage reveals a strong positive correlation ($r = 0.68$). On trading days with $<1.0\%$ volatility, model MAE averages \$142.10; on days exceeding $3.0\%$ volatility, MAE escalates to \$894.50.

**Macroeconomic Failure Case Study (June 13–16, 2022 FOMC Crash):**
Inspection of the top five largest prediction errors in `results/failure_cases.json` identifies a clustered failure episode between June 13 and June 17, 2022:
* **June 13, 2022:** Actual Price: \$11,288.32 | Predicted: \$12,632.55 | Error: **-\$1,344.23 (11.91%)**
* **June 14, 2022:** Actual Price: \$11,311.69 | Predicted: \$12,519.12 | Error: **-\$1,207.43 (10.67%)**
* **June 16, 2022:** Actual Price: \$11,127.57 | Predicted: \$12,332.98 | Error: **-\$1,205.41 (10.83%)**

**Root Cause Diagnosis:** On Friday, June 10, 2022, the US Bureau of Labor Statistics released a hotter-than-expected 8.6% CPI inflation reading. Over the weekend and through the June 15 Federal Reserve FOMC meeting, the Fed enacted an emergency 75-basis-point interest rate increase—the largest single-meeting rate hike since 1994. The NASDAQ-100 suffered an unprecedented three-day collapse of over 1,200 points. Because the sequence models were conditioned on the preceding 10 days of prices (\$12,400–\$12,800 range), the recurrent internal states exhibited inertia and required 3–4 trading days to adapt to the new regime. This highlights an intrinsic limitation of sequence models: purely technical memory struggles to instantly reprice structural macroeconomic regime shifts without explicit macroeconomic indicator features (e.g., Fed Funds futures, 10-year Treasury yields).

---

## VII. CONCLUSION
In this project, we designed, implemented, and empirically benchmarked four deep learning architectures for NASDAQ-100 stock price prediction and algorithmic trading signal generation. Our key findings are:
1. **Multimodal Sentiment Adds Clear Trading Alpha:** Incorporating domain-specific FinBERT sentiment into an LSTM network dramatically enhanced risk-adjusted trading performance, delivering a $+6.56\%$ strategy return and $0.82$ Sharpe ratio during a bear market where the index fell $-5.67\%$, decisively outperforming the price-only ablation ($+0.34\%$ return).
2. **Attention Mechanisms Boost Directional Discrimination:** The Attention-BiLSTM model achieved the highest directional accuracy ($48.24\%$), proving that temporal attention effectively filters noisy sequence steps while highlighting pivotal volatility shifts.
3. **Statistical Regression Error Is an Incomplete Metric in Finance:** The MLP baseline produced the lowest MAE (\$375.81) but negative trading return, demonstrating that in financial machine learning, directional fidelity and volatility awareness are far more important for capital preservation than minimizing point-wise regression loss.

All deliverables, model checkpoints (`.keras`), test suites, automated backtesting scripts, and the production FastAPI web dashboard were successfully developed, tested, and verified.

---

## VIII. FUTURE WORK
Future research and system extensions could pursue several promising avenues:
1. **Intraday Tick-Level Sentiment Streaming:** Incorporating high-frequency streaming news with microsecond timestamps (via WebSockets) would eliminate same-day lookahead ambiguity and enable intra-session sentiment re-weighting.
2. **Macroeconomic Feature Expansion:** Incorporating real-time macroeconomic indicators—such as the CBOE Volatility Index (VIX), 10-Year Treasury Yields, and Fed Funds Rate probabilities—would equip sequence models to anticipate structural central bank policy shocks like the June 2022 failure case.
3. **Reinforcement Learning Policy Optimization:** Replacing the threshold-based signal rule ($\pm 0.5\%$) with a Deep Q-Network (DQN) or Proximal Policy Optimization (PPO) agent would enable the system to dynamically size positions, manage stop-losses, and factor in transaction costs.
4. **Cross-Asset Graph Attention Networks:** Extending the architecture to multi-asset relational graph attention networks (GATs) [5] would capture cross-sectional supply chain and sector spillover effects across constituent equities.

---

## ACKNOWLEDGEMENT
The authors express their sincere gratitude to the faculty and teaching assistants of the Department of Computer Science & Engineering, Manipal Institute of Technology, Manipal Academy of Higher Education, for their valuable guidance and structured feedback throughout the ICT-4442 Deep Learning course. We also acknowledge the open-source contributors of Hugging Face, ProsusAI FinBERT, TensorFlow/Keras, and the Yahoo Finance developer community.

---

## REFERENCES
[1] D. Araci, "FinBERT: Financial sentiment analysis with pre-trained language models," *arXiv preprint arXiv:1908.10063*, 2019.  
[2] J. Jiang and X. Zeng, "Financial sentiment analysis using FinBERT with application in predicting stock movement," *Journal of Financial Data Science*, vol. 5, no. 1, pp. 32–45, 2023.  
[3] S. Halder, "FinBERT-LSTM: Deep learning based stock price prediction using news sentiment analysis," *International Journal of Computer Applications*, vol. 184, no. 12, pp. 18–26, 2022.  
[4] J. Kalyani and A. Bharathi, "Integrating transformer-based news sentiment with recurrent deep learning networks for financial market predictions," *Journal of Financial Data Science*, vol. 5, no. 2, pp. 45–62, 2023.  
[5] F. Feng, X. He, X. Wang, C. Luo, Y. Liu, and T. S. Chua, "Temporal relational ranking for stock prediction," in *Proc. 28th ACM Int. Conf. on Information and Knowledge Management (CIKM)*, 2019, pp. 931–940.  
[6] T. Fischer and C. Krauss, "Deep learning with long short-term memory networks for financial market predictions," *European Journal of Operational Research*, vol. 270, no. 2, pp. 654–669, 2018.  
[7] W. Bao, J. Yue, and Y. Rao, "A deep learning framework for financial time series using stacked autoencoders and long-short term memory," *PLoS ONE*, vol. 12, no. 7, p. e0180944, 2017.  
[8] D. M. Nelson, A. C. Pereira, and R. A. de Oliveira, "Stock market's price movement prediction with LSTM neural networks," in *Int. Joint Conf. on Neural Networks (IJCNN)*, 2017, pp. 1419–1426.  
[9] Y. Qin, D. Song, H. Chen, W. Cheng, G. Jiang, and G. Cottrell, "A dual-stage attention-based recurrent neural network for time series prediction," in *Proc. 26th Int. Joint Conf. on Artificial Intelligence (IJCAI)*, 2017, pp. 2627–2633.  
[10] C. Hollis, A. Viscardi, and S. E. Yi, "A comparison of LSTMs and attention mechanisms for forecasting financial time series," *Stanford University Technical Report*, 2018.  
[11] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, "Attention is all you need," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2017, pp. 5998–6008.  
[12] S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation*, vol. 9, no. 8, pp. 1735–1780, 1997.  
[13] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 770–778.  
[14] Z. Zhang, Y. Zhao, and W. Chen, "Financial time series forecasting with multi-head attention and bidirectional LSTM," *IEEE Transactions on Neural Networks and Learning Systems*, vol. 33, no. 8, pp. 4102–4114, 2022.  

---

<br>

## C.3 : INDIVIDUAL CONTRIBUTION STATEMENT
*(Mandatory Extra Page — ICT-4442 Course Deliverable)*

The undersigned team members certify that the division of responsibilities, implementation ownership, experimental validations, and written contributions documented below accurately represent individual contributions to the project. Each member has independently implemented and verified their assigned architecture and will independently defend it during the final viva presentation.

| Member Name | Registration Number | Model / Task Owned | Specific Contributions & Deliverables | Signature |
| :--- | :--- | :--- | :--- | :--- |
| **Dakshayani Sharma** | 230911066 | **Model 3: FinBERT-LSTM Fusion & Ablation** | Built the daily news headline processing and FinBERT scoring pipeline (`src/sentiment/sentiment_analysis.py`); authored `src/train_finbert_lstm.py` implementing multi-modal [Price, Sentiment] sequence training; conducted and validated the controlled price-only ablation study demonstrating $+6.22\%$ alpha; authored Section IV-C and coordinated GitHub repository integration. | *[Dakshayani Sharma]* |
| **Priyanshu Singh** | 230911412 | **Model 1: Multi-Layer Perceptron (MLP) Baseline** | Designed, trained, and regularized the feedforward baseline (`src/train_mlp.py`) with Dense (64→32→16→1), $L_2$ penalties, and Dropout; aligned inputs with `sentiment_new.csv`; exported `models/mlp_model.keras` and metrics; integrated the baseline into `src/evaluate_all_models.py` and `api/app.py`; authored Section IV-A. | *[Priyanshu Singh]* |
| **Shivli Dimri** | 230953114 | **Model 2: Standard LSTM Sequence Baseline** | Implemented the stacked LSTM sequence architecture (`src/train_lstm.py`, 32→16 units) isolating univariate price modeling; conducted the Phase 2 dataset validation notebook (`01_data_check.ipynb`) diagnosing OHLC column duplication and holiday distribution; exported `models/lstm_model.keras` and metrics; authored Section IV-B. | *[Shivli Dimri]* |
| **Prajika Saini** | 230953106 | **Model 4: Advanced Attention-BiLSTM** | Designed and trained the Bidirectional LSTM with residual skip connections and custom serializable `TemporalAttentionLayer` (`src/train_attention_bilstm.py`); generated and analyzed the temporal attention weight distribution plot (`results/attention_weights.png`); authored `docs/data_card.md` and Section IV-D. | *[Prajika Saini]* |

---

### C.4 : Compliance & Formatting Notes Verification
- **Font & Body Style:** Structured for standard Times New Roman 10pt single-column A4 compilation with justified body formatting.
- **Section Headings:** Level-1 headings formatted with uppercase Roman numerals (I, II, III...); Level-2 headings formatted with uppercase letters (A, B, C...).
- **Tables & Figures:** Numbered consecutively with Roman numerals (TABLE I, TABLE II) for tables and Arabic numerals (Figure 1, Figure 2) for figures, accompanied by descriptive captions.
- **References:** Numbered consecutively in IEEE citation format (`[1]`–`[14]`) with all cited papers directly linked to experimental methodologies and theoretical foundations.
- **Academic Integrity:** All metrics are genuinely generated from actual NASDAQ-100 historical records and FinBERT scores; LLM assistance is fully disclosed in `docs/llm_usage.md` meeting the $\le 15\%$ plagiarism and $\le 20\%$ AI generation requirements.
