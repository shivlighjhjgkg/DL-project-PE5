import argparse
import pandas as pd

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


def FinBERT_sentiment_score(heading):
    """
    Compute sentiment score using pretrained FinBERT
    on a -1 to 1 scale.
    """
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    finbert = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)

    result = nlp(heading)

    if result[0]["label"] == "positive":
        return result[0]["score"]
    elif result[0]["label"] == "neutral":
        return 0.0
    else:
        return -result[0]["score"]


def compute_vader_scores(texts):
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    nltk.download("vader_lexicon", quiet=True)
    analyzer = SentimentIntensityAnalyzer()

    scores = []

    for t in texts:
        if not t:
            scores.append(0.0)
            continue

        r = analyzer.polarity_scores(t)
        scores.append(r.get("compound", 0.0))

    return scores


def compute_finbert_scores(texts):
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    finbert = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)

    scores = []

    for t in texts:
        if not t:
            scores.append(0.0)
            continue

        r = nlp(t)

        lab = r[0]["label"]
        val = r[0]["score"]

        if lab == "positive":
            scores.append(float(val))
        elif lab == "neutral":
            scores.append(0.0)
        else:
            scores.append(-float(val))

    return scores


def main():
    parser = argparse.ArgumentParser(
        description="Compute sentiment per date from news_data.csv"
    )

    parser.add_argument(
        "--method",
        choices=["vader", "finbert"],
        default="finbert",
        help="Which sentiment engine to use"
    )

    parser.add_argument(
        "--infile",
        default="data/raw/news_data.csv",
        help="Input aligned news CSV"
    )

    parser.add_argument(
        "--outfile",
        default="data/processed/sentiment_new.csv",
        help="Output sentiment CSV"
    )

    args = parser.parse_args()

    news_df = pd.read_csv(args.infile)

    texts = []

    for i in range(len(news_df)):
        # Combine non-empty News columns into one string per date
        news_list = news_df.iloc[i, 1:].tolist()
        news_list = [
            str(s).strip()
            for s in news_list
            if pd.notna(s) and str(s).strip() not in {"", "0", "nan", "None"}
        ]
        texts.append(" ".join(news_list))

    if args.method == "finbert":
        scores = compute_finbert_scores(texts)
    else:
        scores = compute_vader_scores(texts)

    news_df["FinBERT score"] = scores

    news_df.to_csv(args.outfile, index=False)

    print(f"Processed dates: {len(news_df)}")
    print(f"Saved to: {args.outfile}")


if __name__ == "__main__":
    main()
