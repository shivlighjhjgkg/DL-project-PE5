import pandas as pd
from pathlib import Path

INPUT_FILE = Path("data/processed/clean_news.csv")
OUTPUT_FILE = Path("data/processed/daily_news.csv")


def aggregate_news(input_file=INPUT_FILE, output_file=OUTPUT_FILE):
    df = pd.read_csv(input_file)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    daily_rows = []

    for date, group in df.groupby("Date"):
        headlines = group["Headline"].tolist()

        row = {"Date": date.strftime("%Y-%m-%d")}

        for i, headline in enumerate(headlines[:10], start=1):
            row[f"News {i}"] = headline

        # Keep the same 10-news-column structure used by
        # the existing FinBERT pipeline.
        for i in range(len(headlines[:10]) + 1, 11):
            row[f"News {i}"] = "0"

        daily_rows.append(row)

    daily_df = pd.DataFrame(daily_rows)

    columns = ["Date"] + [f"News {i}" for i in range(1, 11)]
    daily_df = daily_df[columns]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    daily_df.to_csv(output_file, index=False)

    print(f"Input records: {len(df)}")
    print(f"Output dates: {len(daily_df)}")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    aggregate_news()
