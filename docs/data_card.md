# Dataset Card: NASDAQ-100 (^NDX) & Financial News Sentiment

## 1. Quantitative Market Data
- **Asset:** NASDAQ-100 Index (`^NDX`)
- **Source:** Yahoo Finance (`yfinance`)
- **Date Range:** 2020-10-02 to 2022-09-29 (480 trading days)
- **Features:** Date, Close (Open, High, Low are identical; Volume is unpopulated)
- **Scaling:** MinMax Scaler (fit strictly on the 60% training split)

## 2. Qualitative Financial News Data
- **Source:** Market News feed
- **Date Range:** 2020-10-01 to 2022-09-30 (730 calendar days)
- **Model:** `ProsusAI/finbert` daily scores (`sentiment_new.csv`)

## 3. Merged Data
- **Aligned Records:** 430 trading days after chronological inner join
- **Window Size:** 10-day lookback window predicting next-day close
- **Splits:** Chronological 60% Train / 20% Validation / 20% Test

## 4. Known Data Limitations
- **Single Price Series:** OHLC values are identical, and Volume is zero across all records.
- **Same-Day Leakage Risk:** News headlines lack intraday timestamps, aligning same-day news with same-day close.
- **Short Test Window:** Test split is ~96 trading days.