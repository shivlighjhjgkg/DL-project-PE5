import pandas as pd
from pathlib import Path


INPUT_FILE = Path("data/raw/news.csv")
OUTPUT_FILE = Path("data/processed/clean_news.csv")


def clean_news(input_file=INPUT_FILE, output_file=OUTPUT_FILE):
    df = pd.read_csv(input_file)

    # Normalize dates
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Remove rows with invalid dates
    df = df.dropna(subset=["Date"])

    news_columns = [col for col in df.columns if col.startswith("News ")]

    records = []

    for _, row in df.iterrows():
        date = row["Date"]

        for column in news_columns:
            text = row[column]

            if pd.isna(text):
                continue

            text = str(text).strip()

            # Remove placeholders / empty values
            if not text or text.lower() in {"0", "nan", "none"}:
                continue

            records.append({
                "Date": date,
                "Headline": text
            })

    cleaned = pd.DataFrame(records)

    # Remove exact duplicate date-headline pairs
    cleaned = cleaned.drop_duplicates(
        subset=["Date", "Headline"]
    )

    cleaned = cleaned.sort_values(
        ["Date"]
    ).reset_index(drop=True)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_file, index=False)

    print(f"Input rows: {len(df)}")
    print(f"News records after cleaning: {len(cleaned)}")
    print(f"Unique dates: {cleaned['Date'].nunique()}")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    clean_news()
