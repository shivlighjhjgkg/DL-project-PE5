# Literature Review & Individual Contribution Report

**Student Name:** Priyanshu  
**Project:** FinBERT-LSTM & MLP Multi-Modal Algorithmic Trading System  
**Dataset:** NASDAQ-100 (NDX) Historical Price & Financial News Sentiment  

---

## 1. Literature Review Matrix

| Paper Title & Year | Method / Architecture | Dataset Used | Key Result | Relevance to Our Project |
| :--- | :--- | :--- | :--- | :--- |
| **Araci (2019)**,<br> *"FinBERT: Financial Sentiment Analysis with Pre-trained Language Models"* | Fine-tunes BERT on domain-specific financial text corpora to build **FinBERT**, a contextual language model adapted specifically for financial sentiment analysis. | Two financial sentiment analysis benchmark datasets used to evaluate financial sentiment classification accuracy. | Reports substantial improvements over prior SOTA machine learning methods across all metrics, even with smaller training sets and partial fine-tuning. | **Foundational NLP justification.** Explains and justifies why domain-adapted sentiment scoring (FinBERT) is superior to generic dictionary-based tools (VADER) for financial headlines in our data pipeline. |
| **Jiang and Zeng (2023)**,<br> *"Financial sentiment analysis using FinBERT with application in predicting stock movement"* | Combines FinBERT-derived daily sentiment scores with an LSTM network to forecast price movements; compares against standard BERT, standalone LSTM, and ARIMA models. | Financial news headline corpus paired with historical stock market time series. | Finds that incorporating FinBERT sentiment signals meaningfully improves market fluctuation predictions over standard technical indicators alone. | **Template for multi-modal fusion.** Serves as the blueprint for fusing numerical price histories and NLP sentiment scores into a combined sequential deep learning model. |
| **Halder (2022)**,<br> *"FinBERT-LSTM: Deep Learning based stock price prediction using News Sentiment Analysis"* | Combines domain-specific **FinBERT sentiment embeddings** with a recurrent **LSTM network**; benchmarks against **Multilayer Perceptron (MLP)** and standalone LSTM architectures for next-day close prediction. | Historical price data of the **NASDAQ-100 (NDX)** index paired with financial news articles and headlines collected from *The New York Times*. | Demonstrates that integrating FinBERT sentiment features with sequential price history reduces prediction error (lower MAE and MAPE), while highlighting the sensitivity of deep models to feature scaling and overfitting on small sample sizes. | **Direct architectural baseline for our project.** Our dataset uses the exact NASDAQ-100 (NDX) index and sentiment time series. This paper establishes the benchmark comparisons among MLP baseline, standard LSTM, and FinBERT-LSTM hybrid models implemented in our repository. |
| **Feng et al. (2019)**,<br> *"Temporal Relational Ranking for Stock Prediction"* | Proposes **Relational Stock Ranking (RSR)** utilizing **Temporal Graph Convolution (TGC)** over dynamic price sequences and multi-relational stock graphs (e.g., sector hierarchies, supply-chain links). Optimizes for cross-sectional return ranking. | Historical daily trading data for stocks listed on **NASDAQ and NYSE** coupled with inter-company relational knowledge graphs derived from Wikidata and industry sectors. | Outperforms state-of-the-art sequential models (LSTM, GRU) in investment simulation, demonstrating superior annualized returns and Sharpe ratios by capturing market spillover effects across related corporate entities. | **Theoretical basis for financial backtest metrics.** Justifies evaluating trading systems beyond basic MSE/MAE regression loss, emphasizing risk-adjusted trading metrics (Sharpe ratio, maximum drawdown, cumulative return) as implemented in our backtesting engine. |

---

## 2. Individual Contribution: Multilayer Perceptron (MLP) Baseline & Ensembling

### Overview
My primary contribution to this project is designing, training, and regularizing the **Multilayer Perceptron (MLP) Baseline Model**, evaluating its performance on out-of-sample data, and integrating it into the **Ensemble Strategy** and real-time **API dashboard**.

### Architectural Design
Unlike recurrent networks that maintain internal hidden states over time, the MLP acts as a robust, non-recurrent baseline designed to prevent overfitting on smaller financial time series:

```text
Input (10 timesteps × 2 features: Price + Sentiment)
  │
  ▼
Flatten Layer (20-dimensional feature vector)
  │
  ▼
Dense (64 units, ReLU) + L2 Regularization (λ = 0.001) + Dropout (0.3)
  │
  ▼
Dense (32 units, ReLU) + L2 Regularization (λ = 0.001) + Dropout (0.2)
  │
  ▼
Dense (16 units, ReLU) + L2 Regularization (λ = 0.001)
  │
  ▼
Dense (1 unit, Linear) → Next-Day Predicted Close Price (USD)
```

### Key Contribution Files

1. **`src/train_mlp.py`**
   - Implemented standalone end-to-end training and evaluation script.
   - Incorporated strict chronological train/val/test split (60% / 20% / 20%) to prevent data leakage.
   - Configured `Adam` optimizer ($lr=0.001$), `EarlyStopping` ($patience=25$), and learning rate scheduling (`ReduceLROnPlateau`).
   - Implemented backtesting simulation computing Strategy Return, Buy & Hold Return, Directional Accuracy, and Annualized Sharpe Ratio.

2. **`models/mlp_model.keras` & `api/mlp_model.keras`**
   - Serialized Keras deep learning model checkpoint containing trained weights.

3. **`results/mlp_model_results.png`**
   - Automated visual artifact plotting actual vs. predicted price curves alongside cumulative backtest equity multiples.

4. **`src/evaluate_ensemble.py`**
   - Integrated the MLP baseline into the weighted ensemble model ($0.4 \times \text{MLP} + 0.6 \times \text{LSTM}$), combining simple structural stability with sequential temporal modeling.

5. **`api/app.py`**
   - Integrated model serving endpoints (`/predict` and `/backtest`) supporting `"mlp"` as an independent model choice and rendering it in the interactive web dashboard.

---

## 3. Experimental Results Summary

| Model Architecture | MAE (USD) | MAPE (%) | R² Score | Directional Accuracy | Strategy Return | Sharpe Ratio |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MLP Baseline** | **$392.44** | **3.25%** | **0.5002** | **47.06%** | **+8.49%** | **1.10** |
| Buy & Hold Benchmark | — | — | — | — | -5.67% | -0.21 |
