import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from response_generator import generate_response, generate_chat_response

# Load predefined responses for validation
with open("../data/responses.json", "r") as file:
    predefined_responses = json.load(file)

def test_generate_response():
    """Tests AI response generation for different categories, ensuring policy retrieval and fallback handling."""
    
    test_cases = [
        ("ORDER", "I have an issue with my recent order."),
        ("PAYMENT", "My payment was deducted twice."),
        ("REFUND", "I need a refund."),
        ("FEEDBACK", "Your service is excellent."),
        ("UNKNOWN", "ajdlfkjasdf") 
    ]

    for category, email_text in test_cases:
        response = generate_response(category, email_text)
        
        assert isinstance(response, str) and len(response) > 0, f"Invalid response for {category}"
        
        # If LLM fails, response should be from predefined responses
        if response == "I'm sorry, I don't understand your request.":
            assert category in predefined_responses, f"Missing fallback response for {category}"

    print("Response generation tests passed!")

def test_generate_chat_response():
    """Tests AI chatbot responses with chat history for context."""
    
    chat_history = [
        {"user": "Hi!", "ai": "Hello! How can I assist you today?"},
        {"user": "I need help with my order.", "ai": "Sure! Could you provide more details?"}
    ]
    
    user_message = "It hasn't arrived yet."
    response = generate_chat_response(chat_history, user_message)
    
    assert isinstance(response, str) and len(response) > 0, "Invalid chatbot response"
    
    print("Chatbot response tests passed!")

if __name__ == "__main__":
    test_generate_response()
    test_generate_chat_response()
