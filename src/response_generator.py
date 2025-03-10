from langchain_groq import ChatGroq
from dotenv import load_dotenv
import json
import os
from policy_retriever import retrieve_policy 

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is not set! Please set it in your environment.")

# Load predefined responses as fallback
with open("../data/responses.json", "r") as file:
    responses = json.load(file)

llm = ChatGroq(
    temperature=0.7,  
    groq_api_key=api_key,
    model_name="llama-3.3-70b-versatile"
)
def generate_response(category, email_text):
    """Generates a response using retrieved policy data first, then Llama if needed."""
    try:
        retrieved_policy = retrieve_policy(email_text)

        if retrieved_policy != "NO_MATCH":
            prompt = (
                f"You are an AI assistant for an online shopping platform.\n\n"
                f"Use the following company policy to generate a response:\n\n"
                f"{retrieved_policy}\n\nUser Query: {email_text}"
            )
        else:
            prompt = (
                f"You are an AI assistant for an online shopping platform.\n\n"
                f"Category: {category}\nEmail: {email_text}\nGenerate a response."
            )

        response = llm.invoke(prompt)
        return response.content.strip()

    except Exception as e:
        print(f"Llama Model Error: {e}")

    return responses.get(category, "I'm sorry, I don't understand your request.")

def generate_chat_response(chat_history, user_message):
    """Generates a short, conversational response, prioritizing policy-based answers."""
    try:
        retrieved_policy = retrieve_policy(user_message)

        if retrieved_policy != "NO_MATCH":
            prompt = (
                f"You are an AI assistant for an online shopping platform.\n\n"
                f"User: {user_message}\n\n"
                f"Use the following company policy to provide a conversational response:\n\n"
                f"{retrieved_policy}\n\n"
                f"Respond in a short and conversational manner."
            )
        else:
            chat_context = "\n".join(
                [f"User: {c['user']}\nAI: {c['ai']}" for c in chat_history[-5:]]
            )
            prompt = (
                f"You are an AI assistant for an online shopping platform.\n\n"
                f"{chat_context}\n"
                f"User: {user_message}\n"
                f"AI: Respond in a short and conversational manner."
            )

        response = llm.invoke(prompt)
        return response.content.strip()

    except Exception as e:
        print(f"Llama Model Error: {e}")

    return "I'm here to help! Could you clarify your request?"


def escalate_to_human(category, category_confidence, sentiment, sentiment_confidence):
    """Determines if a case needs to be escalated to human support."""
    
    # Categories that require more human oversight
    CRITICAL_CATEGORIES = {"REFUND", "CANCEL", "PAYMENT"}
    
    if category_confidence < 0.5 and sentiment_confidence < 0.5:
        return True  

    # Escalate if the query falls into a critical category with negative sentiment
    if category in CRITICAL_CATEGORIES and sentiment == "negative":
        return True

    return False


# Example usage
if __name__ == "__main__":
    # Test Email Response
    category = "REFUND"
    email_text = "I ordered a product, but it arrived damaged. I want a full refund."
    print(generate_response(category, email_text))

    # Test Chatbot Response
    chat_history = [
        {"user": "Hi!", "ai": "Hello! How can I assist you today?"},
        {"user": "I need help with my order.", "ai": "Sure! Could you provide more details?"}
    ]
    user_message = "It hasn't arrived yet."
    print(generate_chat_response(chat_history, user_message))
