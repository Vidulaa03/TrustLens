from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

app = Flask(__name__)

# --------------------------------------------------
# Load dataset
# --------------------------------------------------
print("Loading dataset and training TF-IDF model...")

# --------------------------------------------------
# Load and Clean Datasets
# --------------------------------------------------

# -------- NEWS DATASET --------
news = pd.read_csv("dataset/news.csv")
news = news[["text", "label"]]
news["label"] = news["label"].map({"FAKE": 0, "REAL": 1})

# -------- LIAR DATASET --------
# -------- LIAR DATASET --------
liar = pd.read_csv("dataset/train.tsv", sep="\t", header=None)

# Assign correct column names manually
liar.columns = [
    "id", "label", "statement", "subject", "speaker",
    "speaker_job_title", "state_info", "party_affiliation",
    "barely_true_counts", "false_counts",
    "half_true_counts", "mostly_true_counts",
    "pants_on_fire_counts", "context"
]

# Keep only statement and label
liar = liar[["statement", "label"]]

# Convert multi-class labels to binary
fake_labels = ["pants-fire", "false", "barely-true"]
real_labels = ["half-true", "mostly-true", "true"]

liar["label"] = liar["label"].apply(
    lambda x: 0 if x in fake_labels else 1
)

# Rename statement column
liar = liar.rename(columns={"statement": "text"})
liar = liar[["text", "label"]]

fake_labels = ["pants-fire", "false", "barely-true"]
real_labels = ["half-true", "mostly-true", "true"]

liar["label"] = liar["label"].apply(
    lambda x: 0 if x in fake_labels else 1
)

liar = liar.rename(columns={"statement": "text"})
liar = liar[["text", "label"]]

# -------- MULTIFC DATASET --------
# multifc = pd.read_csv("dataset/train.csv")

# # Keep only claim and label
# multifc = multifc[["claim", "label"]]

# # Convert labels to binary
# fake_labels = ["false", "pants-fire", "barely-true"]
# real_labels = ["true", "mostly-true", "half-true"]

# multifc["label"] = multifc["label"].apply(
#     lambda x: 0 if str(x).lower() in fake_labels else 1
# )

# Rename claim column to text
# multifc = multifc.rename(columns={"claim": "text"})
# multifc = multifc[["text", "label"]]


# -------- COMBINE ALL --------
combined = pd.concat([news, liar], ignore_index=True)
combined.dropna(inplace=True)

print("Total samples after merging:", len(combined))
print(combined["label"].value_counts())

X = combined["text"]
y = combined["label"]

# -----------------------------
# Train-Test Split (IMPORTANT)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Logistic Regression model
model = LogisticRegression(class_weight="balanced", max_iter=1000)
model.fit(X_train_tfidf, y_train)

# -----------------------------
# Model Evaluation
# -----------------------------
y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)



print("\n📊 Model Performance:")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall:", round(recall * 100, 2), "%")
print("F1 Score:", round(f1 * 100, 2), "%")

print("\nDetailed Report:\n")
print(classification_report(y_test, y_pred))

feature_names = vectorizer.get_feature_names_out()

print("Model ready 🚀")

# --------------------------------------------------
# Helper function
# --------------------------------------------------
def analyze_news(text):

    text_tfidf = vectorizer.transform([text])

    probs = model.predict_proba(text_tfidf)[0]
    fake_prob = probs[0]
    real_prob = probs[1]

    score = round(max(fake_prob, real_prob) * 100, 2)
    label = "Real News" if real_prob > fake_prob else "Fake News"

    coefs = model.coef_[0]
    tfidf_array = text_tfidf.toarray()[0]

    important_indices = np.argsort(tfidf_array * coefs)[-5:]
    keywords = [feature_names[i] for i in important_indices if tfidf_array[i] > 0]

    keyword_text = ", ".join(keywords) if keywords else "linguistic patterns"

    if label == "Real News":
        explanation = (
            "The article uses neutral and factual language, avoids sensational phrasing, "
            "and follows linguistic patterns commonly found in verified news sources. "
            "Key indicators such as " + keyword_text +
            " suggest authentic reporting styles."
        )
    else:
        explanation = (
            "The article contains emotionally charged language and stylistic patterns "
            "frequently observed in misleading news. Terms such as " + keyword_text +
            " reflect low-credibility linguistic features."
        )

    return label, score, explanation

# --------------------------------------------------
# Routes
# --------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    score = None
    explanation = None
    news_text = ""

    if request.method == "POST":
        news_text = request.form["news"]
        result, score, explanation = analyze_news(news_text)

    return render_template(
        "index.html",
        result=result,
        score=score,
        explanation=explanation,
        news_text=news_text
    )

# --------------------------------------------------
# Run App
# --------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)


