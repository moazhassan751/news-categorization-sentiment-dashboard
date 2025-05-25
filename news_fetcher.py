import json
from newsapi import NewsApiClient
from datetime import datetime

# Initialize NewsAPI
NEWSAPI_KEY = "08d49c2d053a4b5a88f187ac1b099523"
newsapi = NewsApiClient(api_key=NEWSAPI_KEY)

# List of countries to fetch news for
COUNTRIES = ["us", "gb", "in", "au", "ca", "de", "fr"]
# Output JSON file
OUTPUT_FILE = "news_data.json"

def fetch_latest_headlines(country: str, page_size: int = 10):
    try:
        # Fetch top headlines for the given country
        resp = newsapi.get_top_headlines(country=country, page_size=page_size)
        print(f"[{country}] status={resp.get('status')}, totalResults={resp.get('totalResults')}")

        if resp.get("totalResults", 0) > 0:
            return resp["articles"]

        # Fallback to sources if no articles are found
        srcs = newsapi.get_sources(country=country).get("sources", [])
        if not srcs:
            return []
        ids = [s["id"] for s in srcs]
        resp2 = newsapi.get_top_headlines(sources=",".join(ids), page_size=page_size)
        print(f"[{country}-fallback] totalResults={resp2.get('totalResults')}")
        return resp2.get("articles", [])
    except Exception as e:
        print(f"Error fetching headlines for {country}: {e}")
        return []

def save_news_to_json(countries, page_size=10):
    news_data = {"header": {"timestamp": datetime.now().isoformat(), "countries": countries}, "news": []}

    for country in countries:
        print(f"Fetching news for {country.upper()}...")
        articles = fetch_latest_headlines(country, page_size)
        print(f"Fetched {len(articles)} articles for {country.upper()}")
        for article in articles:
            news_data["news"].append({
                "country": country,
                "title": article.get("title"),
                "description": article.get("description"),
                "url": article.get("url"),
                "publishedAt": article.get("publishedAt"),
                "source": article.get("source", {}).get("name")
            })

    # Save the news data to a JSON file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(news_data, f, ensure_ascii=False, indent=4)
    print(f"News data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    save_news_to_json(COUNTRIES, page_size=10)
