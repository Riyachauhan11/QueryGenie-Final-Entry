import os
import pickle
import numpy as np

# Get absolute paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
MODEL_PATH = os.path.join(BASE_DIR, "models/email_classifier.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models/vectorizer_email.pkl")

# Load trained model
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)


except Exception as e:
    print(f"Error loading model/vectorizer: {e}")
    exit()

# Function to classify user query 
def classify_email(user_query):
    """Classify customer query into predefined categories with confidence score."""
    try:
        query_vec = vectorizer.transform([user_query])
        prediction = model.predict(query_vec)[0]

        proba = model.predict_proba(query_vec)[0]
        confidence = np.max(proba) 

        return prediction, confidence
    except Exception as e:
        print(f"Error classifying query: {e}")
        return None, 0.0

if __name__ == "__main__":
    sample_query = "I want to cancel my subscription. How can I do that?"
    category, confidence = classify_email(sample_query)
    print(f"Predicted Category: {category} (Confidence: {confidence:.2f})")
