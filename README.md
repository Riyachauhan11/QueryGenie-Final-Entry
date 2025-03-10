# QueryGenie - 2025 Girl Hackathon

QueryGenie is an AI-powered customer support automation system that categorizes incoming queries and emails, analyzes sentiment, and generates AI-driven responses based on confidence scores and the urgency or intensity of emotions detected in customer requests.

> 📌 **Tech Stack Used**  
> - **Programming Language:** Python  
> - **Machine Learning & NLP:** Scikit-learn, NLTK, Transformers (Hugging Face)  
> - **Data Handling:** Pandas, Pickle  
> - **Frontend:** Streamlit  
> - **Environment & Config Management:** Dotenv  

**Note:** This project is a **prototype** designed to fulfill the high-level functionality of the final product. It is will undergo significant changes in future iterations.

## 📁 Project Structure
```
📂 QueryGenie
 ├── 📂 data  # Contains datasets for classification & sentiment analysis
 │   ├── emails.csv  # Customer support dataset 
 │   ├── sentiment_data.csv  # sentiment dataset 
 │   ├── responses.json  # Predefined fallback responses for LLM failures
 │
 ├── 📂 models  # Pre-trained models for classification & sentiment analysis
 │   ├── email_classifier.pkl
 │   ├── sentiment_analyzer.pkl
 │   ├── vectorizer_email.pkl
 │   ├── vectorizer_sentiment.pkl
 │
 ├── 📂 src  # Core system logic
 │   ├── 📂 training  # Scripts for training ML models
 │   │   ├── train_classifier.py
 │   │   ├── train_sentiment.py
 │   ├── sentiment_analysis.py  # Performs sentiment analysis on queries
 │   ├── classification.py  # Categorizes emails into predefined types
 │   ├── response_generator.py  # Generates AI-based responses
 │   ├── main.py  # Main execution script (Streamlit frontend)
 │
 ├── 📂 tests  # Unit test scripts
 │   ├── test_classification.py
 │   ├── test_sentiment.py
 │   ├── test_response.py
 │
 ├── .env  # Stores API keys (Groq API Key required)
 ├── .gitignore  # Specifies ignored files
 ├── README.md  # Project documentation
 ├── requirements.txt  # Lists required dependencies
```

# 🛠️ Setup Instructions  

## 1️⃣ Clone the Repository  

First, clone this repository to your local machine:  

```bash
git clone https://github.com/Riyachauhan11/QueryGenie.git
cd QueryGenie
```

---

## 2️⃣ Install Dependencies  

Ensure you have **Python 3.8+** installed. Then, install the required dependencies:  

```bash
pip install -r requirements.txt
```

🔹 **(Optional) Using a Virtual Environment:**  
If you prefer **isolating dependencies**, set up a virtual environment:  

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

---

## 3️⃣ Set Up Environment Variables  

This project uses **Groq's Llama model** for AI-generated responses. You need an API key from **[GroqCloud](https://console.groq.com/playground)**:  

🔹 **Steps to Get Groq API Key:**  
1. **Sign up** at **[GroqCloud](https://console.groq.com/playground)** and log in.  
2. **Navigate to API Keys** in your account settings.  
3. **Generate a new API key** and copy it.  
4. **Create a `.env` file** in the project root and add:  

   ```plaintext
   GROQ_API_KEY=your_api_key_here
   ```

---

## 4️⃣ Include Datasets in the `data` Folder  

Ensure the `data` folder contains the necessary datasets for **email classification** and **sentiment analysis**:  

🔹 **Customer Query Classification Dataset:**  
Used to categorize customer queries into predefined types.  
📌 [Download from Kaggle](https://www.kaggle.com/datasets/scodepy/customer-support-intent-dataset)  

🔹 **Sentiment Analysis Dataset:**  
Used to analyze if a customer's email sentiment is positive or negative.  
📌 [Download from Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)  

💾 **Steps:**  
1. Download both datasets as **CSV files**.  
2. Place them inside the **`data`** folder following the project structure.  

Even though responses are generated using an **LLM**, a fallback mechanism **`responses.json`** has also been set up. It contains predefined responses to be used when the LLM faces issues generating responses.  

---

## 5️⃣ Training Models (Optional)  

If the pre-trained models (`email_classifier.pkl`, `sentiment_analyzer.pkl`) do not load or crash, retrain them using the scripts in the `src/training` folder:  

```bash
python src/training/train_classifier.py  # Train email classification model
python src/training/train_sentiment.py  # Train sentiment analysis model
```

This will regenerate the `.pkl` model files in the `models` directory.  

---

## 6️⃣ Run Tests in PowerShell  

To ensure everything works correctly, run unit tests before launching the project:  

```bash
cd tests
python test_classification.py  # Test email classification
python test_sentiment.py       # Test sentiment analysis
python test_response.py        # Test response generation
```

If all tests pass, proceed to the next step.  

---

## 7️⃣ Running the Project  

The frontend is built using **Streamlit**. To launch the app:  

```bash
cd src
streamlit run main.py
```

This will start the application and open the **QueryGenie UI** in your browser, where you can enter customer queries and receive AI-generated responses. How the UI looks like:
![image](https://github.com/user-attachments/assets/8ebb3c3b-ac34-4fc5-97a4-0b4c478cb84c)
  
---

## How QueryGenie Works?  

1. **Process Incoming Query**  
   - A customer query is received and processed using trained models.  
   - The **email classifier** identifies the type of query.  
   - The **sentiment analyzer** determines if the sentiment is **positive** or **negative**.  

2. **Query Categorization**  
   - The classifier uses a dataset (`emails.csv`) to match the query with a predefined category (e.g., Order, Refund, Payment, etc.).  

3. **Sentiment Analysis**  
   - The sentiment model checks the emotional tone of the query using the sentiment dataset (`sentiment_data.csv`).  

4. **Confidence Score Evaluation**  
   - Both classification and sentiment models return a **confidence score**.  
   - If the confidence score is **high**, the AI-generated response is **accepted** and sent to the customer.  

5. **Escalation to Human Agent**  
   - If the sentiment is **negative** and the query falls into a **sensitive category** (e.g., Refund, Complaint), the request is **forwarded to a human agent** for review.

---

## 🔹 Final Notes  

- This project is a **prototype** and not the final version.  
- The datasets used for email classification and sentiment analysis are **not extensive or broad enough** to cover every possible customer query. A more **comprehensive dataset** would be needed for real-world deployment.  
- The system is specifically designed for handling **customer support queries** and may **not work** for other domains.  
- Since the datasets are limited, the model **may not always produce accurate results** and might misclassify some queries.  
- For response generation, a **general LLM** has been used. However, in a real-world scenario, an **LLM trained on company policies and guidelines** would be used to ensure **accurate and policy-compliant** responses.

