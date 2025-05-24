# categorizer.py

import joblib
from preprocessor import TextPreprocessor
import json

# load the pipeline (which uses TextPreprocessor internally)
category_model = joblib.load('best_category_model.pkl')

def predict_category(text: str) -> str:
    return category_model.predict([text])[0]

# Input JSON file (news data without categories)
INPUT_FILE = "news_data.json"

# Output JSON file (news data with predicted categories)
OUTPUT_FILE = "categorized_news.json"

def categorize_news(input_file: str, output_file: str):
    try:
        # Load the news data from the input JSON file
        with open(input_file, "r", encoding="utf-8") as f:
            news_data = json.load(f)

        # Ensure the "news" section exists in the JSON
        if "news" not in news_data:
            print("No news data found in the input file.")
            return

        # Add predicted categories to each news article
        for article in news_data["news"]:
            headline = article.get("title", "")
            if headline:
                # Predict the category using the categorizer model
                article["predicted_category"] = predict_category(headline)
            else:
                article["predicted_category"] = "Unknown"

        # Save the updated news data to the output JSON file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(news_data, f, ensure_ascii=False, indent=4)

        print(f"Categorized news data saved to {output_file}")

    except Exception as e:
        print(f"Error categorizing news: {e}")

if __name__ == "__main__":
    categorize_news(INPUT_FILE, OUTPUT_FILE)
