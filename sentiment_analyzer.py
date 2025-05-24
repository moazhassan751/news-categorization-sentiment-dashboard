import json
import joblib
from preprocessor import TextPreprocessor

# Load the sentiment model
sentiment_model = joblib.load('best_sentiment_model.pkl')

def analyze_sentiment(text: str) -> str:
    if not text:
        return 'Neutral'
    return sentiment_model.predict([text])[0]

# Input and output JSON file (same file is updated)
INPUT_OUTPUT_FILE = "categorized_news.json"

def add_sentiment_to_news(input_output_file: str):
    try:
        # Load the categorized news data
        with open(input_output_file, "r", encoding="utf-8") as f:
            news_data = json.load(f)

        # Ensure the "news" section exists in the JSON
        if "news" not in news_data:
            print("No news data found in the input file.")
            return

        # Add sentiment to each news article
        for article in news_data["news"]:
            headline = article.get("title", "")
            if headline:
                # Analyze the sentiment of the headline
                article["sentiment"] = analyze_sentiment(headline)
            else:
                article["sentiment"] = "Neutral"

        # Save the updated news data back to the same JSON file
        with open(input_output_file, "w", encoding="utf-8") as f:
            json.dump(news_data, f, ensure_ascii=False, indent=4)

        print(f"Sentiment added to news data and saved to {input_output_file}")

    except Exception as e:
        print(f"Error adding sentiment to news: {e}")

if __name__ == "__main__":
    add_sentiment_to_news(INPUT_OUTPUT_FILE)
