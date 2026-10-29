# Disclosure of Large Language Model (LLM) Usage
**Course:** ICT-4442 Deep Learning Project  
**Team 12:** FinBERT-LSTM Quantitative Stock Prediction & Trading System  
**Members:**  
- Dakshayani Sharma (230911066)
- Priyanshu Singh (230911412)
- Shivli Dimri (230953114)
- Prajika Saini (230953106)

---

## 1. Compliance Statement
In strict adherence to Section 9 (*Academic Integrity*) of the ICT 4442 Course Guidelines, this document explicitly details all instances where Large Language Models (LLMs) were consulted during the development of this project.

All model implementations, architectural designs, experimental metrics, plots, and analysis reflect authentic data execution. LLMs were employed exclusively as an assistive pair-programming and formatting tool under direct student direction and review, ensuring compliance with the $<15\%$ plagiarism and $<20\%$ generated content benchmarks.

---

## 2. Granular Record of LLM Assistance

| Phase / Module | File / Component | Nature of LLM Usage | Student Verification & Contribution |
| :--- | :--- | :--- | :--- |
| **Phase 1: Synopsis** | Preliminary literature search | Assisted in identifying relevant citation formats for foundational papers (Araci, Feng, Vaswani). | Formulated core hypotheses, selected the 4 target architectures, and defined objectives. |
| **Phase 2: Data Preprocessing** | `src/data/data_cleaning.py` & `src/sentiment/sentiment_analysis.py` | Assisted in debugging date formatting inconsistencies between UTC news timestamps and NASDAQ trading days. | Identified the absence of authentic OHLC variance in `stock_price.csv` and authored the data check notebook (`01_data_check.ipynb`). |
| **Phase 3: Baseline Modeling** | `src/train_mlp.py` | Assisted in refactoring the Keras model definition to support L2 regularization and clean JSON metrics serialization. | Priyanshu designed the feedforward layer dimensions (64→32→16→1), early stopping patience, and evaluation logic. |
| **Phase 3: Sequential Modeling** | `src/train_lstm.py` | Assisted in diagnosing macOS sandbox cache permission issues (`KERAS_HOME`, `MPLCONFIGDIR`). | Shivli isolated the sequential price-only baseline (32→16 stacked LSTM) and verified chronological out-of-sample scaling. |
| **Phase 3: Multimodal Fusion** | `src/train_finbert_lstm.py` | Assisted in structuring the parallel ablation harness comparing multimodal (+sentiment) vs price-only runs. | Dakshayani engineered the multi-feature windowing pipeline and analyzed the Sharpe ratio delta (+0.82 vs +0.18). |
| **Phase 3: Attention Mechanism** | `src/train_attention_bilstm.py` | Assisted in implementing the custom Keras serializable `TemporalAttentionLayer` and extracting attention weights. | Prajika designed the Bidirectional LSTM + residual skip connection topology and verified temporal weights. |
| **Phase 3: Unified Benchmarking** | `src/evaluate_all_models.py` | Assisted in assembling matplotlib grid subplots for Q-Q error distributions and failure case tables. | Verified all evaluation metrics across the test split, confirmed absence of lookahead bias, and calculated drawdown. |
| **Phase 3: API & Dashboard** | `api/app.py` | Assisted in writing Chart.js frontend configuration and FastAPI Pydantic schema validation. | Integrated model checkpoints, tested REST endpoints (`/health`, `/predict`, `/api/backtest/data`), and verified responses. |
| **Phase 3: Report Authoring** | `docs/final_report_part_c.md` | Assisted in formatting IEEE citations and standardizing section headers according to Part C guidelines. | Authored all technical descriptions, architectural justifications, error analysis narratives, and conclusion. |

---

## 3. Human Authorship & Verification Guarantee
- **No Fabricated Data or Metrics:** Every metric reported in the final comparison table (MAE, RMSE, MAPE, $R^2$, Directional Accuracy, Strategy Return, Sharpe Ratio, Win Rate) was calculated programmatically from model inference over the unseen NASDAQ-100 test set (`2022-05-20` to `2022-09-29`).
- **Defensibility in Viva:** Each team member has individually designed and validated their assigned model code and is fully prepared to defend the architecture, mathematics, and empirical results during the in-person presentation and Q&A.
