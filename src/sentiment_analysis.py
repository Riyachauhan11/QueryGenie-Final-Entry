import os
import pickle
import numpy as np

# Get absolute paths for models
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
MODEL_PATH = os.path.join(BASE_DIR, "models/sentiment_analyzer.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models/vectorizer_sentiment.pkl")

# Load sentiment analysis model
try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    with open(VECTORIZER_PATH, "rb") as file:
        vectorizer = pickle.load(file)

except FileNotFoundError:
    print("Error: Model files not found. Ensure 'sentiment_analyzer.pkl' & 'vectorizer_sentiment.pkl' exist in 'models/' folder.")
    exit()

def analyze_sentiment(text, confidence_threshold=0.65):
    """Predict sentiment (positive, neutral, negative) with confidence check."""
    if not text.strip():
        return "neutral", 0.0  # Handle empty input

    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]

    # Get probability scores
    proba = model.predict_proba(text_vec)[0]
    confidence = round(np.max(proba), 2) 

    # Force "neutral" if confidence is below threshold
    if confidence < confidence_threshold:
        return "neutral", confidence

    return prediction, confidence

if __name__ == "__main__":
    # Test cases
    test_reviews = [
        "The service was terrible. I want a refund.",
        "I'm happy with my purchase, great quality!"
    ]

    for review in test_reviews:
        sentiment, confidence = analyze_sentiment(review)
        print(f"\nReview: {review if review else '(Empty Input)'}")
        print(f"Sentiment: {sentiment} (Confidence: {confidence})")
