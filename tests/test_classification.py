import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from classification import classify_email  

def test_email_classification():
    """Tests email classification for different types of queries."""
    
    test_cases = [
        ("I want to cancel my order.", ["ORDER"]),
        ("I need a refund for my damaged product.", ["REFUND"]),
        ("My payment was deducted twice. Please help.", ["PAYMENT"]),
        ("Can you provide me an invoice for the purchased product?", ["INVOICE","ORDER"]),
        ("I want to cancel my subscription.", ["SUBSCRIPTION", "CANCEL"]),
        ("Where is my package? I haven't received it yet.", ["DELIVERY", "SHIPPING"]),
        ("hHelp me file a consumer complaint.", ["FEEDBACK"]),
        ("I want to contact your support team.", ["CONTACT"]),
    ]

    for email_text, expected_categories in test_cases:
        category, confidence = classify_email(email_text)
        
        assert category is not None, f"Category is None for input: {email_text}"
        assert category in expected_categories, f"Unexpected category {category} for input: {email_text}"
        assert confidence is not None and 0.0 <= confidence <= 1.0, f"Invalid confidence {confidence} for input: {email_text}"

    print("Email classification tests passed!")

if __name__ == "__main__":
    test_email_classification()
