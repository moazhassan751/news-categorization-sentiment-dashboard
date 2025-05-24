import streamlit as st
import pandas as pd
import json
from datetime import datetime
from news_fetcher import save_news_to_json
from categorizer import categorize_news
from sentiment_analyzer import add_sentiment_to_news
import plotly.express as px

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="News Categorization & Sentiment Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DATA PREP ---
def fetch_categorize_analyze_news():
    save_news_to_json(["us", "gb", "in", "au", "ca", "de", "fr"], page_size=50)
    categorize_news("news_data.json", "categorized_news.json")
    add_sentiment_to_news("categorized_news.json")

def load_categorized_news():
    try:
        with open("categorized_news.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading categorized news: {e}")
        return {"news": []}

fetch_categorize_analyze_news()
news_data = load_categorized_news()
df = pd.DataFrame(news_data["news"])

# --- DATA CLEANING ---
if df.empty or "country" not in df.columns:
    st.error("No news data available or missing 'country' field in the data.")
    st.stop()

df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors='coerce')
df = df.dropna(subset=["publishedAt"])

# --- SIDEBAR FILTERS ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/6/6a/Newspaper_icon.png", width=80)
st.sidebar.title("Filters")

countries = sorted(df["country"].dropna().unique().tolist())
categories = sorted(df["predicted_category"].dropna().unique().tolist())
sentiments = sorted(df["sentiment"].dropna().unique().tolist())

selected_countries = st.sidebar.multiselect("Country", countries, default=countries)
selected_categories = st.sidebar.multiselect("Category", categories, default=categories)
selected_sentiments = st.sidebar.multiselect("Sentiment", sentiments, default=sentiments)

min_date = df["publishedAt"].min().date()
max_date = df["publishedAt"].max().date()
date_range = st.sidebar.date_input(
    "Published Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)
if isinstance(date_range, tuple):
    start_date, end_date = date_range
else:
    start_date = end_date = date_range

st.sidebar.markdown("---")
st.sidebar.caption("Powered by NewsAPI, ML Categorizer & Sentiment Model")

# --- FILTER DATA ---
filtered_df = df[
    df["country"].isin(selected_countries) &
    df["predicted_category"].isin(selected_categories) &
    df["sentiment"].isin(selected_sentiments) &
    (df["publishedAt"].dt.date >= start_date) &
    (df["publishedAt"].dt.date <= end_date)
]

# --- HEADER ---
st.markdown(
    """
    <div style='display:flex;align-items:center;gap:20px;'>
        <img src='https://upload.wikimedia.org/wikipedia/commons/6/6a/Newspaper_icon.png' width='60'/>
        <div>
            <h1 style='margin-bottom:0;'>🗞 News Categorization & Sentiment Dashboard</h1>
            <span style='font-size:18px;color:#444;'>Live news, categorized and analyzed for sentiment.<br>
            Use the filters to explore by country, category, sentiment, and date.</span>
        </div>
    </div>
    """, unsafe_allow_html=True
)
st.markdown("---")

# --- KPIs ---
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("📰 Total Articles", len(filtered_df))
kpi2.metric("📂 Categories", filtered_df["predicted_category"].nunique())
kpi3.metric("💬 Sentiments", filtered_df["sentiment"].nunique())

st.markdown("")

# --- GRAPHS ---
if not filtered_df.empty:
    col_g1, col_g2 = st.columns([2,1])
    with col_g1:
        cat_counts = filtered_df["predicted_category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        fig_cat = px.bar(
            cat_counts, x="Category", y="Count", color="Category",
            title="Articles by Category", text_auto=True, color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_cat.update_layout(
            xaxis_title="Category",
            yaxis_title="Number of Articles",
            plot_bgcolor="#000000",
            showlegend=False,
            margin=dict(l=10, r=10, t=30, b=10)
        )
        st.plotly_chart(fig_cat, use_container_width=True)
    with col_g2:
        sent_counts = filtered_df["sentiment"].value_counts().reset_index()
        sent_counts.columns = ["Sentiment", "Count"]
        fig_sent = px.pie(
            sent_counts, names="Sentiment", values="Count",
            title="Sentiment Distribution", hole=0.5,
            color_discrete_map={"Positive":"#2ecc40", "Negative":"#ff4136", "Neutral":"#ffdc00"}
        )
        fig_sent.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_sent, use_container_width=True)

    # Time series
    st.markdown("#### Articles Over Time")
    time_counts = filtered_df.groupby(filtered_df["publishedAt"].dt.date).size().reset_index(name='Articles')
    fig_time = px.line(
        time_counts, x="publishedAt", y="Articles",
        markers=True, title="", labels={"publishedAt": "Date", "Articles": "Number of Articles"},
        color_discrete_sequence=["#0072C6"]
    )
    fig_time.update_layout(plot_bgcolor="#f8f9fa", margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_time, use_container_width=True)
else:
    st.info("No data available for the selected filters.")

st.markdown("---")

# --- ARTICLE CARDS ---
st.markdown("### 📰 Articles")
def sentiment_color(sentiment):
    if sentiment.lower() == "positive":
        return "#2ecc40"
    elif sentiment.lower() == "negative":
        return "#ff4136"
    else:
        return "#ffdc00"

if not filtered_df.empty:
    for _, row in filtered_df.iterrows():
        headline = row['title'] if pd.notnull(row['title']) and str(row['title']).strip() else f"[No Headline] ({row['predicted_category']})"
        st.markdown(
            f"""
            <div style='background:#f8f9fa;padding:18px 18px 8px 18px;margin-bottom:18px;border-radius:10px;box-shadow:0 2px 8px #eee;'>
                <h4 style='margin-bottom:0;color:#222;'>{headline}</h4>
                <div style='font-size:13px;color:#888;margin-bottom:6px;'>
                    <b>Published:</b> {row['publishedAt'].strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp;
                    <b>Category:</b> <span style='color:#0072C6'>{row['predicted_category']}</span> &nbsp;|&nbsp;
                    <b>Sentiment:</b> <span style='color:{sentiment_color(row['sentiment'])};font-weight:bold'>{row['sentiment']}</span> &nbsp;|&nbsp;
                    <b>Source:</b> {row['source']}
                </div>
                <div style='font-size:15px;color:#333;margin-bottom:8px;'>{row['description']}</div>
                <a href='{row['url']}' target='_blank' style='color:#0056b3;font-weight:bold;'>Read Full Article</a>
            </div>
            """, unsafe_allow_html=True
        )
else:
    st.warning("No articles to display for the selected filters.")