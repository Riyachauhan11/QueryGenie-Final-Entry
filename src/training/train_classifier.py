import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Get absolute paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_PATH = os.path.join(BASE_DIR, "data/emails.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models/email_classifier.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models/vectorizer_email.pkl")

try:
    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Ensure required columns exist
    if "instruction" not in df.columns or "category" not in df.columns:
        raise ValueError("Dataset is missing required columns: 'instruction' or 'category'.")

    # Drop any missing values
    df = df.dropna(subset=["instruction", "category"])

    print(f"Loaded dataset: {df.shape[0]} rows")
    print("🔹 Category Distribution:\n", df["category"].value_counts())

except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Train TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(df["instruction"])
y_train = df["category"]

# Train Naive Bayes Model
model = MultinomialNB()
model.fit(X_train, y_train)

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Save trained model & vectorizer
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

with open(VECTORIZER_PATH, "wb") as f:
    pickle.dump(vectorizer, f)

print("Email classifier trained and saved successfully!")
