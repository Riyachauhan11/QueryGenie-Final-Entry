import streamlit as st
import time
from classification import classify_email
from sentiment_analysis import analyze_sentiment
from response_generator import generate_response, escalate_to_human, generate_chat_response

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "latest_response" not in st.session_state:
    st.session_state.latest_response = None
if "latest_feedback" not in st.session_state:
    st.session_state.latest_feedback = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("📧 QueryGenie - AI Customer Support")
st.markdown("Simulate AI-powered email support, engage in chatbot conversations, and access past email history logs with QueryGenie.")

# Tabs for Email, Chatbot & History
tabs = st.tabs(["📩 Send Email", "💬 Chatbot", "📜 Email History"])

# --- EMAIL PROCESSING TAB ---

with tabs[0]:
    st.subheader("📩 AI Email Support")
    st.write("Send an email query and get AI-powered responses to your queries with intelligent classification and sentiment analysis.")
    email_subject = st.text_input("📌 Subject", "")
    email_text = st.text_area("✉️ Email Body", "")

    if st.button("📤 Send & Process"):
        if email_text.strip():
            start_time = time.time()

            # AI Processing
            category, category_confidence = classify_email(email_text)
            sentiment, sentiment_confidence = analyze_sentiment(email_text)
            response = generate_response(category, email_text)
            escalation_needed = escalate_to_human(category, category_confidence, sentiment, sentiment_confidence)

            response_time = time.time() - start_time

            # Store latest response
            st.session_state.latest_response = {
                "subject": email_subject or "No Subject",
                "email": email_text,
                "category": category,
                "category_confidence": category_confidence,
                "sentiment": sentiment,
                "sentiment_confidence": sentiment_confidence,
                "response": response,
                "escalation": escalation_needed,
                "time": response_time,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            st.session_state.latest_feedback = None  # Reset feedback
            st.session_state.history.append(st.session_state.latest_response)

    # Display Latest Response
    if st.session_state.latest_response:
        entry = st.session_state.latest_response
        st.subheader("📊 AI Response")
        st.write(f"📌 **Category:** {entry['category']} (Confidence: {entry['category_confidence']:.2f})")
        st.write(f"🔍 **Sentiment:** {entry['sentiment']} (Confidence: {entry['sentiment_confidence']:.2f})")
        st.write(f"⏳ **Response Time:** {entry['time']:.2f} sec")
        st.write(f"💬 **AI Response:** {entry['response']}")

        if entry["escalation"]:
            st.error("⚠️ This email requires escalation to a human agent!")
        else:
            st.success("✅ AI response is sufficient.")

        # Feedback Buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✔️ Helpful", key="helpful"):
                st.session_state.latest_feedback = "helpful"
        with col2:
            if st.button("❌ Not Helpful", key="not_helpful"):
                st.session_state.latest_feedback = "not helpful"

        if st.session_state.latest_feedback:
            if st.session_state.latest_feedback == "helpful":
                st.success("✔️ Thank you for your feedback!")
            else:
                st.warning("❌ We'll work on improving future responses!")


# --- CHATBOT TAB ---
with tabs[1]:
    st.subheader("💬 AI Chatbot")
    st.write("Chat with our AI support assistant. For the best results, provide descriptive questions for more accurate responses.")

    chat_input = st.text_input("🗨️ Type your message:")

    if st.button("💬 Send", key="send_chat"):
        if chat_input.strip():
            # Use the last 5 messages as context for the response
            recent_history = st.session_state.chat_history[-5:]
            bot_response = generate_chat_response(recent_history, chat_input)

            # Store conversation history
            st.session_state.chat_history.append({"user": chat_input, "ai": bot_response})

    # Display Chat History
    for chat in reversed(st.session_state.chat_history):
        with st.container():
            st.markdown(f"👤 **You:** {chat['user']}")
            st.markdown(f"🤖 **AI:** {chat['ai']}")
            st.markdown("---")

# --- HISTORY TAB ---
with tabs[2]:
    st.subheader("📜 Email History")
    if st.session_state.history:
        for i, entry in enumerate(reversed(st.session_state.history)):
            with st.expander(f"📧 {entry['subject']} ({entry['timestamp']})"):
                st.write(f"✉️ **Email:** {entry['email']}")
                st.write(f"📌 **Category:** {entry['category']} (Confidence: {entry['category_confidence']:.2f})")
                st.write(f"🔍 **Sentiment:** {entry['sentiment']} (Confidence: {entry['sentiment_confidence']:.2f})")
                st.write(f"⏳ **Response Time:** {entry['time']:.2f} sec")
                st.write(f"💬 **AI Response:** {entry['response']}")

                if entry["escalation"]:
                    st.error("⚠️ Escalated to a human agent")
    else:
        st.info("No history yet. Send an email to see past responses!")
