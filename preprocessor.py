import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK data is available
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

def preprocess_text(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'\d+', '', text)                      # Remove numbers
    text = re.sub(r'[^\w\s]', ' ', text)                 # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()             # Remove extra spaces
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]  # Remove stopwords and short tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t, pos='v') for t in tokens]         # Lemmatize as verb
    tokens = [lemmatizer.lemmatize(t, pos='n') for t in tokens]         # Lemmatize as noun
    return ' '.join(tokens)

from sklearn.base import BaseEstimator, TransformerMixin

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return [preprocess_text(text) for text in X]