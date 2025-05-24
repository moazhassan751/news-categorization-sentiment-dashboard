import nltk
import pandas as pd
import os
import sys
import warnings
import joblib

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import FunctionTransformer

warnings.filterwarnings('ignore')

# Import your custom preprocessor
try:
    from preprocessor import TextPreprocessor
except ImportError:
    print("Error: 'preprocessor.py' with class 'TextPreprocessor' is required.")
    sys.exit(1)

def load_and_split_data():
    """Load and split both datasets."""
    if not os.path.exists('NewsCategorizer.csv'):
        print("Error: 'NewsCategorizer.csv' not found.")
        sys.exit(1)
    if not os.path.exists('news_sentiment_merged.csv'):
        print("Error: 'news_sentiment_merged.csv' not found.")
        sys.exit(1)

    cat_df = pd.read_csv('NewsCategorizer.csv')
    sent_df = pd.read_csv('news_sentiment_merged.csv').dropna(subset=['Sentiment'])

    X_cat = cat_df['headline']
    y_cat = cat_df['category']

    X_sent = sent_df['Headlines']
    y_sent = sent_df['Sentiment']

    X_cat_train, X_cat_test, y_cat_train, y_cat_test = train_test_split(X_cat, y_cat, test_size=0.2, random_state=42)
    X_sent_train, X_sent_test, y_sent_train, y_sent_test = train_test_split(X_sent, y_sent, test_size=0.2, random_state=42)

    return X_cat_train, X_cat_test, y_cat_train, y_cat_test, X_sent_train, X_sent_test, y_sent_train, y_sent_test

def train_and_evaluate_models(X_train, y_train, X_test, y_test, task_name):
    models = {
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': MultinomialNB(),
        'Linear SVM': LinearSVC(max_iter=1000),
        'Logistic Regression': LogisticRegression(max_iter=1000)
    }

    best_model = None
    best_accuracy = 0
    best_name = ""

    preprocessor = TextPreprocessor()

    print(f"\n=== {task_name} Model Comparison ===\n")

    for name, clf in models.items():
        pipeline = Pipeline([
            ('preprocess', FunctionTransformer(preprocessor.transform, validate=False)),
            ('tfidf', TfidfVectorizer(max_features=10000)),
            ('clf', clf)
        ])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        acc = accuracy_score(y_test, preds)

        print(f"{name} Accuracy: {acc:.4f}")
        print(classification_report(y_test, preds))

        if acc > best_accuracy:
            best_accuracy = acc
            best_model = pipeline
            best_name = name

    print(f"\nBest Model for {task_name}: {best_name} with Accuracy {best_accuracy:.4f}\n")
    return best_model

def main():
    (X_cat_train, X_cat_test, y_cat_train, y_cat_test,
     X_sent_train, X_sent_test, y_sent_train, y_sent_test) = load_and_split_data()

    # News Categorization
    best_cat_model = train_and_evaluate_models(
        X_cat_train, y_cat_train, X_cat_test, y_cat_test, "News Categorization"
    )
    joblib.dump(best_cat_model, 'best_category_model.pkl')

    # Sentiment Analysis
    best_sent_model = train_and_evaluate_models(
        X_sent_train, y_sent_train, X_sent_test, y_sent_test, "Sentiment Analysis"
    )
    joblib.dump(best_sent_model, 'best_sentiment_model.pkl')

    # Prediction demo
    new_headline = "Apple launches new iPhone with innovative features"
    category = best_cat_model.predict([new_headline])[0]
    sentiment = best_sent_model.predict([new_headline])[0]

    print("\n=== Example Prediction ===")
    print(f"Headline: {new_headline}")
    print(f"Predicted Category: {category}")
    print(f"Predicted Sentiment: {sentiment}")

if __name__ == "__main__":
    main()