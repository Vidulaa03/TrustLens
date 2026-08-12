# from flask import Flask, render_template, request
# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# app = Flask(__name__)
# analyzer = SentimentIntensityAnalyzer()

# # ---------------- MODEL SETUP ----------------

# news = pd.read_csv("dataset/news.csv")
# news = news[["text", "label"]]
# news["label"] = news["label"].map({"FAKE": 0, "REAL": 1})

# X = news["text"]
# y = news["label"]

# vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
# X_tfidf = vectorizer.fit_transform(X)

# model = LogisticRegression(max_iter=1000)
# model.fit(X_tfidf, y)

# feature_names = vectorizer.get_feature_names_out()

# # ---------------- ANALYSIS FUNCTION ----------------

# def analyze_news(text):

#     text_tfidf = vectorizer.transform([text])
#     probs = model.predict_proba(text_tfidf)[0]

#     fake_prob = probs[0]
#     real_prob = probs[1]

#     sentiment = analyzer.polarity_scores(text)
#     emotional_score = round((sentiment["pos"] + sentiment["neg"]) * 100, 2)

#     clickbait_score = min(text.count("!") * 5, 100)

#     trust_score = round(
#         (real_prob * 100)
#         - (emotional_score * 0.15)
#         - (clickbait_score * 0.3),
#         2
#     )

#     trust_score = max(min(trust_score, 100), 0)

#     label = "Real News" if real_prob > fake_prob else "Fake News"

#     return {
#         "label": label,
#         "trust_score": trust_score,
#         "emotional_score": emotional_score,
#         "clickbait_score": clickbait_score
#     }

# # ---------------- ROUTES ----------------

# @app.route("/")
# def dashboard_page():
#     return render_template("dashboard.html")

# @app.route("/analyze", methods=["GET", "POST"])
# def analyze_page():

#     analysis = None
#     news_text = ""

#     if request.method == "POST":
#         news_text = request.form["news"]
#         analysis = analyze_news(news_text)

#     return render_template(
#         "analyze.html",
#         analysis=analysis,
#         news_text=news_text
#     )

# if __name__ == "__main__":
#     app.run(debug=True)

# ===============================
# ADD THIS TO YOUR OLD app.py
# Do NOT replace model section
# Paste below analyze_news()
# ===============================

from __future__ import annotations

import csv
import io
from datetime import datetime
from functools import wraps

from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from models.database import (
    clear_history,
    create_user,
    delete_analysis,
    fetch_all_analyses,
    get_user_by_id,
    get_user_by_username_or_email,
    init_db,
    insert_analysis,
)
from services.ai_detection_service import analyze_ai_signals
from services.analytics_service import build_dashboard_summary, build_statistics_payload, build_trends_payload, classify_risk
from services.bias_service import analyze_bias
from services.credibility_service import analyze_news
from services.headline_service import analyze_headline_consistency
from services.topic_service import classify_topic
from utils.text_utils import character_count, clean_text, word_count

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False
app.secret_key = "trustlens-dev-secret-key"
init_db()


def _auth_response(success, message=None, user=None):
    return {"success": success, "message": message, "user": user}


def _current_user():
    user_id = session.get("user_id")
    if not user_id:
        return None
    return get_user_by_id(user_id)


def _login_user(user):
    session["user_id"] = user["id"]
    session["user_name"] = user["name"]
    session["user_username"] = user["username"]


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if not _current_user():
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    return wrapped_view


@app.route("/api/auth/register", methods=["POST"])
def api_register():
    payload = request.get_json(silent=True) or {}
    name = (payload.get("name") or "").strip()
    username = (payload.get("username") or "").strip()
    email = (payload.get("email") or "").strip()
    password = payload.get("password") or ""

    if not name or not username or not email or not password:
        return jsonify(_auth_response(False, "Name, username, email, and password are required."))

    if len(password) < 6:
        return jsonify(_auth_response(False, "Password must be at least 6 characters long."))

    if get_user_by_username_or_email(username) or get_user_by_username_or_email(email):
        return jsonify(_auth_response(False, "An account with that username or email already exists."))

    user_id = create_user(name, username, email, generate_password_hash(password))
    user = get_user_by_id(user_id)
    _login_user(user)
    return jsonify(_auth_response(True, "User registered successfully.", {"id": user["id"], "name": user["name"], "username": user["username"], "email": user["email"]}))


@app.route("/api/auth/login", methods=["POST"])
def api_login():
    payload = request.get_json(silent=True) or {}
    identifier = (payload.get("username") or payload.get("email") or "").strip()
    password = payload.get("password") or ""

    if not identifier or not password:
        return jsonify(_auth_response(False, "Username/email and password are required."))

    user = get_user_by_username_or_email(identifier)
    if not user or not check_password_hash(user["password_hash"], password):
        return jsonify(_auth_response(False, "Invalid username/email or password."))

    _login_user(user)
    return jsonify(_auth_response(True, "Login successful.", {"id": user["id"], "name": user["name"], "username": user["username"], "email": user["email"]}))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""
        user = get_user_by_username_or_email(username)
        if user and check_password_hash(user["password_hash"], password):
            _login_user(user)
            return redirect(url_for("dashboard"))
        return render_template("login.html", error="Invalid username/email or password.")
    return render_template("login.html", error=None)


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = (request.form.get("name") or "").strip()
        username = (request.form.get("username") or "").strip()
        email = (request.form.get("email") or "").strip()
        password = request.form.get("password") or ""

        if not name or not username or not email or not password:
            return render_template("register.html", error="Please fill in all fields.")
        if len(password) < 6:
            return render_template("register.html", error="Password must be at least 6 characters long.")
        if get_user_by_username_or_email(username) or get_user_by_username_or_email(email):
            return render_template("register.html", error="That username or email is already in use.")

        user_id = create_user(name, username, email, generate_password_hash(password))
        user = get_user_by_id(user_id)
        _login_user(user)
        return redirect(url_for("dashboard"))

    return render_template("register.html", error=None)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


def _json_response(success, data=None, error=None):
    return {"success": success, "data": data, "error": error}


def _record_from_analysis(article_text, headline, analysis, risk_level=None):
    return {
        "article_text": clean_text(article_text)[:5000],
        "headline": clean_text(headline)[:500] if headline else "",
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "trust_score": int(analysis.get("trust_score", 0)),
        "fake_probability": float(analysis.get("fake_probability", 0)),
        "real_probability": float(analysis.get("real_probability", 0)),
        "bias_score": float(analysis.get("bias_score", 0)),
        "ai_probability": float(analysis.get("ai_probability", 0)),
        "topic": analysis.get("topic") or "General",
        "headline_score": int(analysis.get("headline_score", 50)),
        "risk_level": risk_level or analysis.get("risk_level") or "LOW",
    }


def _save_analysis_record(article_text, headline, analysis):
    risk_level = classify_risk(
        analysis.get("trust_score", 0),
        analysis.get("bias_score", 0),
        analysis.get("clickbait_score", 0),
        analysis.get("headline_score", 50),
        analysis.get("ai_probability", 0),
    )
    analysis["risk_level"] = risk_level
    insert_analysis(_record_from_analysis(article_text, headline, analysis, risk_level))
    return analysis


@app.route("/")
def index():
    if _current_user():
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    records = fetch_all_analyses()
    summary = build_dashboard_summary(records)
    return render_template("dashboard.html", dashboard_data=summary)


@app.route("/analyze", methods=["GET", "POST"])
@login_required
def analyze():
    article_text = ""
    analysis = None
    error = None

    if request.method == "POST":
        if request.is_json:
            payload = request.get_json(silent=True) or {}
            article_text = payload.get("article_text") or payload.get("text") or ""
        else:
            article_text = request.form.get("article_text") or request.form.get("text") or ""

        article_text = clean_text(article_text)
        if not article_text:
            error = "Article text is required."
        else:
            try:
                analysis = analyze_news(article_text)
                analysis = _save_analysis_record(article_text, None, analysis)
            except ValueError as exc:
                error = str(exc)

        if request.is_json:
            return jsonify(_json_response(success=error is None, data=analysis, error=error))

    return render_template("analyze.html", article_text=article_text, analysis=analysis, error=error)


@app.route("/history")
@login_required
def history():
    records = fetch_all_analyses()
    search = request.args.get("search", "").strip().lower()
    topic = request.args.get("topic", "").strip()
    risk = request.args.get("risk", "").strip()
    score = request.args.get("score", "").strip()

    filtered = []
    for row in records:
        article = (row["article_text"] or "").lower()
        if search and search not in article:
            continue
        if topic and row.get("topic") != topic:
            continue
        if risk and row.get("risk_level") != risk:
            continue
        if score:
            target = int(score)
            if row.get("trust_score", 0) < target:
                continue
        filtered.append(row)

    topics = sorted({row.get("topic") or "General" for row in records})
    risks = sorted({row.get("risk_level") or "LOW" for row in records})
    return render_template("history.html", analyses=filtered, topics=topics, risks=risks)


@app.route("/history/delete/<int:analysis_id>", methods=["POST"])
@login_required
def delete_history_entry(analysis_id):
    delete_analysis(analysis_id)
    return redirect(url_for("history"))


@app.route("/history/export.csv")
@login_required
def export_history_csv():
    analyses = fetch_all_analyses()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "date", "article", "topic", "trust_score", "bias_score", "ai_probability", "risk_level"])
    for row in analyses:
        writer.writerow([
            row["id"],
            row["timestamp"],
            (row["article_text"] or "")[:120],
            row.get("topic") or "General",
            row.get("trust_score"),
            row.get("bias_score"),
            row.get("ai_probability"),
            row.get("risk_level"),
        ])
    csv_data = output.getvalue()
    return csv_data, 200, {"Content-Type": "text/csv", "Content-Disposition": "attachment; filename=trustlens_history.csv"}


@app.route("/bias-detection", methods=["GET", "POST"])
def bias_detection():
    return redirect(url_for("analyze"))


@app.route("/topic-classification", methods=["GET", "POST"])
def topic_classification():
    return redirect(url_for("analyze"))


@app.route("/headline-consistency", methods=["GET", "POST"])
def headline_consistency():
    return redirect(url_for("analyze"))


@app.route("/ai-detection", methods=["GET", "POST"])
def ai_detection():
    return redirect(url_for("analyze"))


@app.route("/statistics")
@login_required
def statistics():
    records = fetch_all_analyses()
    if not records:
        return render_template("statistics.html", stats=None)
    stats = build_statistics_payload(records)
    return render_template("statistics.html", stats=stats)


@app.route("/trends")
@login_required
def trends():
    records = fetch_all_analyses()
    if not records:
        return render_template("trends.html", trend_data=None)
    trend_data = build_trends_payload(records)
    return render_template("trends.html", trend_data=trend_data)


@app.route("/settings", methods=["GET", "POST"])
@login_required
def settings():
    if request.method == "POST":
        if request.form.get("action") == "clear_history":
            clear_history()
            return redirect(url_for("settings"))
    return render_template("settings.html")


@app.route("/api-access")
@login_required
def api_access():
    return render_template("api_access.html", api_status="Connected", usage_total=0, usage_limit=1000)


@app.route("/profile")
@login_required
def profile():
    user = _current_user()
    records = fetch_all_analyses()
    return render_template(
        "profile.html",
        user=user,
        analysis_count=len(records),
        recent_analysis=records[:3] if records else [],
    )


if __name__ == "__main__":
    app.run(debug=True)