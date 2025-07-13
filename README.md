# 📊 Stock & News Sentiment Analysis with Transformers

This repository contains hands-on code and focused on **sentiment analysis using the Transformers library**.

We applied a fine-tuned BERT model from HuggingFace — [`slisowski/stock_sentiment_hp`](https://huggingface.co/slisowski/stock_sentiment_hp) — specifically built for sentiment classification in **stock market and financial news datasets**.

This project showcases how to preprocess raw text and analyze sentiment from real-world financial and news data — all in Python using HuggingFace Transformers.

---

## 🔧 Tech Stack

- 🐍 Python 3.x  
- 🤗 Transformers (`pipeline`)  
- 🧠 Model: [`slisowski/stock_sentiment_hp`](https://huggingface.co/slisowski/stock_sentiment_hp)  
- 📚 NLP Preprocessing: NLTK  
- 📊 Data Handling: Pandas  
- ✅ Evaluation: scikit-learn

---

## 🚀 Features

- Load and clean real datasets (stock headlines and article snippets)
- Apply core NLP preprocessing steps:
  - Tokenization
  - Stopword removal
  - Lemmatization
- Perform sentiment analysis using a Transformer model
- Visualize and evaluate the results using a confusion matrix and classification report

---
🧼 Text Preprocessing
We applied the following steps to clean and normalize the text:

Tokenization using NLTK

> Stopword Removal using nltk.corpus.stopwords
> Lemmatization using WordNetLemmatizer
Example:
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered = [t for t in tokens if t not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(t) for t in filtered]
    return ' '.join(lemmatized)

📊 Example Output:
| News/Article Headline                                  | Sentiment | Confidence |
| ------------------------------------------------------ | --------- | ---------- |
| "Apple hits record high after earnings beat estimates" | Positive  | 0.97       |
| "Company struggles amid weak global demand"            | Negative  | 0.94       |
| "Market expected to remain steady this week"           | Neutral   | 0.87       |


📈 Model Evaluation (Optional)
If your dataset contains ground truth labels (Positive, Negative, etc.), you can evaluate accuracy using:
from sklearn.metrics import classification_report

print(classification_report(df["actual"], df["predicted"]))
▶️ How to Run
Clone the Repository

bash
Copy
Edit
git clone https://github.com/your-username/stock-news-sentiment-analysis.git
cd stock-news-sentiment-analysis
Install Required Libraries

bash
Copy
Edit
pip install -r requirements.txt
Run the Jupyter Notebook

Open sentiment_analysis.ipynb in Jupyter Notebook or VS Code and execute step by step.

🔗 HuggingFace Model Info
We used the slisowski/stock_sentiment_hp model:

Trained on: financial and stock-market texts

Base model: BERT

Tasks: Sentiment classification (positive, neutral, negative)

Model page: View on HuggingFace

🧠 Model by @slisowski via HuggingFace

📊 Dataset inspired by public financial and news headlines

💬 Questions or Feedback?
Connect with me on LinkedIn or drop a comment in the repository.


