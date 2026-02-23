from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = Flask(__name__)
analyzer = SentimentIntensityAnalyzer()

# ---------------- MODEL SETUP ----------------

news = pd.read_csv("dataset/news.csv")
news = news[["text", "label"]]
news["label"] = news["label"].map({"FAKE": 0, "REAL": 1})

X = news["text"]
y = news["label"]

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_tfidf, y)

feature_names = vectorizer.get_feature_names_out()

# ---------------- ANALYSIS FUNCTION ----------------

def analyze_news(text):

    text_tfidf = vectorizer.transform([text])
    probs = model.predict_proba(text_tfidf)[0]

    fake_prob = probs[0]
    real_prob = probs[1]

    sentiment = analyzer.polarity_scores(text)
    emotional_score = round((sentiment["pos"] + sentiment["neg"]) * 100, 2)

    clickbait_score = min(text.count("!") * 5, 100)

    trust_score = round(
        (real_prob * 100)
        - (emotional_score * 0.15)
        - (clickbait_score * 0.3),
        2
    )

    trust_score = max(min(trust_score, 100), 0)

    label = "Real News" if real_prob > fake_prob else "Fake News"

    return {
        "label": label,
        "trust_score": trust_score,
        "emotional_score": emotional_score,
        "clickbait_score": clickbait_score
    }

# ---------------- ROUTES ----------------

@app.route("/")
def dashboard_page():
    return render_template("dashboard.html")

@app.route("/analyze", methods=["GET", "POST"])
def analyze_page():

    analysis = None
    news_text = ""

    if request.method == "POST":
        news_text = request.form["news"]
        analysis = analyze_news(news_text)

    return render_template(
        "analyze.html",
        analysis=analysis,
        news_text=news_text
    )

if __name__ == "__main__":
    app.run(debug=True)

