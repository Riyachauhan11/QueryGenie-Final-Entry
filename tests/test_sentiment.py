import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from sentiment_analysis import analyze_sentiment

def test_analyze_sentiment():
    """Tests sentiment analysis for different types of input."""
    
    test_cases = [
        ("I love this service!", "positive"),
        ("This is the worst experience ever.", "negative"),
        ("The product is okay, nothing special.", "neutral"),
        ("", "neutral"), 
    ]

    for text, expected_sentiment in test_cases:
        sentiment, confidence = analyze_sentiment(text)
        assert sentiment == expected_sentiment, f"Expected {expected_sentiment} but got {sentiment} for input: {text}"
        assert 0.0 <= confidence <= 1.0, f"Confidence out of range for input: {text}"

    print("Sentiment analysis tests passed!")

if __name__ == "__main__":
    test_analyze_sentiment()
