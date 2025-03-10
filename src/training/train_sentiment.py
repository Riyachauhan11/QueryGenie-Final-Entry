import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Get absolute path 
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_PATH = os.path.join(BASE_DIR, "data/sentiment_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models/sentiment_analyzer.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models/vectorizer_sentiment.pkl")

try:
    # Load TweetEval Sentiment Dataset
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={"label": "sentiment", "text": "text"})

    # Map sentiment labels to "negative", "neutral", "positive"
    sentiment_mapping = {0: "negative", 1: "neutral", 2: "positive"}
    df["sentiment"] = df["sentiment"].map(sentiment_mapping)

    # Drop any missing values
    df = df.dropna()

    print(f"Loaded dataset: {df.shape[0]} rows")
    print("🔹 Sentiment Distribution:\n", df["sentiment"].value_counts())

except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Train model
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))  
X_train = vectorizer.fit_transform(df["text"])
y_train = df["sentiment"]

model = LogisticRegression(max_iter=300, class_weight="balanced")  
model.fit(X_train, y_train)

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Save trained model & vectorizer
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

with open(VECTORIZER_PATH, "wb") as f:
    pickle.dump(vectorizer, f)

print("Sentiment model trained and saved successfully!")
