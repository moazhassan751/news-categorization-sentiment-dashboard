# News Categorization & Sentiment Dashboard

A Streamlit dashboard for live news aggregation, automatic categorization, and sentiment analysis. The app fetches news from multiple countries, classifies articles into categories, analyzes sentiment, and provides interactive visualizations and filters.

## Features

- **Live News Fetching:** Aggregates news from multiple countries using NewsAPI.
- **Automatic Categorization:** Classifies articles into categories using a machine learning model.
- **Sentiment Analysis:** Assigns sentiment (Positive, Negative, Neutral) to each article.
- **Interactive Dashboard:** 
  - Filter by country, category, sentiment, and date.
  - View KPIs (total articles, categories, sentiments).
  - Visualize articles by category, sentiment distribution, and time series.
  - Browse articles with metadata and direct links.

## Project Structure

```
app.py
categorizer.py
sentiment_analyzer.py
news_fetcher.py
preprocessor.py
train_models.py
utils.py
categorized_news.json
news_data.json
best_category_model.pkl
best_sentiment_model.pkl
NewsCategorizer.csv
news_sentiment_merged.csv
gnews_multi_country_news.json
```

## Getting Started

### Prerequisites

- Python 3.8+
- [Streamlit](https://streamlit.io/)
- [pandas](https://pandas.pydata.org/)
- [plotly](https://plotly.com/python/)
- Other dependencies: see below

### Installation

1. **Clone the repository:**
    ```sh
    git clone <your-repo-url>
    cd <repo-folder>
    ```

2. **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```
    If `requirements.txt` is missing, install manually:
    ```sh
    pip install streamlit pandas plotly scikit-learn
    ```

3. **Download or place your NewsAPI key** (if required by `news_fetcher.py`).

4. **(Optional) Train or update models:**
    - Use `train_models.py` to retrain category/sentiment models if needed.

## Running the App

```sh
streamlit run app.py
```

The dashboard will open in your browser.

## File Descriptions

- **app.py**: Main Streamlit dashboard application.
- **news_fetcher.py**: Fetches news articles from NewsAPI and saves to JSON.
- **categorizer.py**: Loads ML model to categorize news articles.
- **sentiment_analyzer.py**: Loads ML model to assign sentiment to articles.
- **preprocessor.py**: Text preprocessing utilities.
- **train_models.py**: Scripts to train category and sentiment models.
- **utils.py**: Utility functions.
- **best_category_model.pkl / best_sentiment_model.pkl**: Pre-trained ML models.
- **categorized_news.json / news_data.json**: News data files.
- **NewsCategorizer.csv / news_sentiment_merged.csv**: Datasets for training/evaluation.

## Customization

- **Add/Remove Countries:** Edit the country list in `app.py` or `news_fetcher.py`.
- **Change Categories:** Update the model or mapping in `categorizer.py`.
- **Improve Sentiment Analysis:** Retrain the sentiment model with more data.


## Credits

- NewsAPI for news data
- Plotly for visualizations
- Streamlit for dashboard UI
