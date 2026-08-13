import pickle
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from services.ai_detection_service import calculate_ai_probability
from services.topic_service import classify_topic
from utils.text_utils import clean_text

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except Exception:
    SentimentIntensityAnalyzer = None

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
DATASET_PATH = Path(__file__).resolve().parent.parent / "dataset" / "news.csv"

vectorizer = None
model = None
analyzer = SentimentIntensityAnalyzer() if SentimentIntensityAnalyzer else None


def _train_model_assets():
    global vectorizer, model

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    if "title" in df.columns and "text" in df.columns:
        X = df["title"].fillna("") + " " + df["text"].fillna("")
    elif "text" in df.columns:
        X = df["text"].fillna("")
    else:
        raise ValueError("Dataset must contain either a text column or title+text columns.")

    y = df["label"]

    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
    X_tfidf = vectorizer.fit_transform(X.astype(str))

    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X_tfidf, y)

    vector_path = MODEL_DIR / "tfidf.pkl"
    model_path = MODEL_DIR / "model.pkl"
    with open(vector_path, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)


def _load_model_assets():
    global vectorizer, model
    if vectorizer is not None and model is not None:
        return

    vector_path = MODEL_DIR / "tfidf.pkl"
    model_path = MODEL_DIR / "model.pkl"

    if vector_path.exists() and model_path.exists():
        try:
            with open(vector_path, "rb") as f:
                vectorizer = pickle.load(f)
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            return
        except Exception:
            vectorizer = None
            model = None

    _train_model_assets()


def _article_language_signals(cleaned, clickbait_score, emotional_score):
    lower = cleaned.lower()
    political_hits = [
        word for word in [
            "election", "campaign", "government", "policy", "party", "senate",
            "congress", "parliament", "vote", "debate", "minister", "president",
            "candidate", "coalition", "politics", "democrat", "republican", "opposition"
        ] if word in lower
    ]
    sensational_hits = [
        word for word in [
            "shocking", "secret", "bombshell", "revealed", "breaking", "must", "outrage",
            "hidden", "exposed", "panic", "disaster", "unbelievable", "dramatic"
        ] if word in lower
    ]
    one_sided_hits = [
        word for word in [
            "obviously", "clearly", "everyone", "nobody", "always", "never", "simply",
            "frankly", "without question", "obvious", "totally"
        ] if word in lower
    ]

    political_score = min(100, max(0, len(set(political_hits)) * 14 + (1 if "policy" in lower or "government" in lower else 0) * 12))
    sensational_score = min(100, max(0, clickbait_score + len(set(sensational_hits)) * 12 + cleaned.count("!") * 5))
    one_sided_score = min(100, max(0, len(set(one_sided_hits)) * 18 + cleaned.count("!") * 3 + len(re.findall(r"\b(always|never|everyone|nobody|obviously|clearly)\b", lower)) * 10))
    emotional_score = min(100, max(0, emotional_score))

    return {
        "political_score": round(political_score, 2),
        "sensational_score": round(sensational_score, 2),
        "one_sided_score": round(one_sided_score, 2),
        "emotional_score": round(emotional_score, 2),
    }


def analyze_news(text):
    _load_model_assets()
    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("Article text is required.")
    if len(cleaned.split()) < 20:
        raise ValueError("Please provide at least 20 words for a meaningful analysis.")

    if vectorizer is not None and model is not None:
        text_tfidf = vectorizer.transform([cleaned])
        probabilities = model.predict_proba(text_tfidf)[0]
        fake_prob = float(probabilities[0])
        real_prob = float(probabilities[1])
    else:
        word_count = len(re.findall(r"\b\w+\b", cleaned))
        subjectivity = min(1.0, max(0.1, word_count / 700.0))
        sentiment_bias = 0.62 if "election" in cleaned.lower() or "government" in cleaned.lower() else 0.55
        fake_prob = max(0.05, min(0.95, (1 - subjectivity) * sentiment_bias))
        real_prob = 1.0 - fake_prob

    sentiment = _sentiment_scores(cleaned)
    emotional_score = round((sentiment["pos"] + sentiment["neg"]) * 100, 2)
    clickbait_score = min(100, max(0, round((cleaned.count("!") * 6) + (sum(1 for word in ["shocking", "secret", "revealed", "must", "outrage", "breaking", "bombshell"] if word in cleaned.lower()) * 9)) ))
    language_signals = _article_language_signals(cleaned, clickbait_score, emotional_score)

    ai_probability = calculate_ai_probability(cleaned)
    topic_result = classify_topic(cleaned)
    trust_score = max(0, min(100, round((real_prob * 100) - (language_signals["emotional_score"] * 0.18) - (clickbait_score * 0.25) + (topic_result["confidence"] * 0.1), 2)))

    label = "Real News" if real_prob >= fake_prob else "Fake News"
    risk_level = "HIGH" if trust_score < 45 or clickbait_score > 65 or ai_probability > 75 else "MEDIUM" if trust_score < 70 or clickbait_score > 35 else "LOW"

    return {
        "label": label,
        "trust_score": int(round(trust_score)),
        "fake_probability": round(fake_prob * 100, 2),
        "real_probability": round(real_prob * 100, 2),
        "emotional_score": language_signals["emotional_score"],
        "political_score": language_signals["political_score"],
        "sensational_score": language_signals["sensational_score"],
        "one_sided_score": language_signals["one_sided_score"],
        "clickbait_score": round(clickbait_score, 2),
        "ai_probability": ai_probability,
        "topic": topic_result["primary_topic"],
        "headline_score": max(0, min(100, 100 - clickbait_score)),
        "risk_level": risk_level,
        "bias_score": round(min(100, max(0, language_signals["emotional_score"] * 0.9 + clickbait_score * 0.6 + language_signals["one_sided_score"] * 0.3)), 2),
        "signals_summary": (
            "Detected emotional, political, sensational, and one-sided language patterns in the article. "
            "AI-writing indicators and narrative framing are combined into the final trust signal."
        ),
    }


def _sentiment_scores(text):
    if analyzer is None:
        positive = sum(1 for word in ["good", "positive", "strong", "support", "improve", "trust", "benefit", "hope"] if word in text.lower())
        negative = sum(1 for word in ["bad", "negative", "fraud", "scandal", "corrupt", "outrage", "shocking", "collapse", "lie", "fear"] if word in text.lower())
        total = max(1, positive + negative)
        return {"pos": positive / total, "neg": negative / total}

    scores = analyzer.polarity_scores(text)
    return {
        "pos": scores.get("pos", 0.0),
        "neg": scores.get("neg", 0.0),
    }
