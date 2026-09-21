import pandas as pd

news_df = pd.read_csv("data/raw/news.csv")

stock_df = pd.read_csv("data/raw/stock_price.csv")

# Normalize stock dates to YYYY-MM-DD in a vectorized, safe way to avoid
# chained-assignment / SettingWithCopyWarning and future pandas changes.
# The stock file contains timezone-aware timestamps like
# "2020-10-01 00:00:00-04:00" so parse as datetimes and format as dates.
# Some timestamps include timezone offsets (e.g. '2020-10-01 00:00:00-04:00').
# To avoid mixed time zone parsing issues, trim the timezone suffix first
# and then parse the naive timestamp portion.

stock_df["Date"] = pd.to_datetime(
    stock_df["Date"].astype(str).str[:19],
    errors="coerce"
).dt.strftime("%Y-%m-%d")

# Keep only news rows that have dates present in the stock data
news_df = news_df[news_df["Date"].isin(stock_df["Date"].tolist())]

news_df.to_csv("data/raw/news_data.csv", index=False)
