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

import base64
import csv
import io
import re
from datetime import datetime
from html.parser import HTMLParser
from functools import wraps
from urllib.parse import urlparse
from urllib.request import Request, urlopen

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
    recalculate_risk_levels,
)
from services.ai_detection_service import analyze_ai_signals
from services.analytics_service import build_dashboard_summary, build_statistics_payload, build_trends_payload, classify_risk
from services.enhanced_analytics import (
    build_dashboard_summary_v2,
    build_statistics_payload_v2,
    build_trends_payload_v2,
    calculate_risk_distribution,
    calculate_risk_level,
    format_date_short,
    generate_intelligence_summary,
    generate_why_flagged_summary,
    generate_recommended_action,
    get_trust_score_interpretation,
)
from services.bias_service import analyze_bias
from services.credibility_service import analyze_news
from services.headline_service import analyze_headline_consistency
from services.topic_service import classify_topic
from utils.text_utils import character_count, clean_text, word_count

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False
app.secret_key = "trustlens-dev-secret-key"
init_db()
recalculate_risk_levels()


class _ArticleTextParser(HTMLParser):
    """Extract visible text without making article analysis depend on bs4."""

    _ignored_tags = {"script", "style", "noscript", "svg", "template"}

    def __init__(self):
        super().__init__()
        self.parts = []
        self._ignored_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag.lower() in self._ignored_tags:
            self._ignored_depth += 1

    def handle_endtag(self, tag):
        if tag.lower() in self._ignored_tags and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data):
        if not self._ignored_depth:
            self.parts.append(data)

    def text(self):
        return clean_text(" ".join(self.parts))


def _article_text_from_response(text, content_type=""):
    """Convert an HTML response into visible text; leave text responses intact."""
    if content_type in {"text/html", "application/xhtml+xml"} or re.search(r"<html[\s>]", text, re.I):
        parser = _ArticleTextParser()
        parser.feed(text)
        text = parser.text()
    if re.search(r"<!doctype\s+html|<html[\s>]", text, re.I):
        raise ValueError("The URL did not provide readable article text.")
    return text


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
    risk_level = calculate_risk_level(
        analysis.get("trust_score", 0),
        analysis.get("bias_score", 0),
        analysis.get("clickbait_score", 0),
        analysis.get("headline_score", 50),
        analysis.get("ai_probability", 0),
    )
    analysis["risk_level"] = risk_level
    analysis_id = insert_analysis(_record_from_analysis(article_text, headline, analysis, risk_level))
    analysis["id"] = analysis_id
    return analysis


def _read_url_text(url_value):
    if not url_value:
        raise ValueError("URL is required.")

    raw_url = url_value.strip()
    if raw_url.startswith("data:"):
        header, _, payload = raw_url.partition(",")
        if not payload:
            raise ValueError("The provided data URL is empty.")
        if ";base64" in header.lower():
            try:
                text = base64.b64decode(payload).decode("utf-8", errors="replace")
            except Exception as exc:
                raise ValueError("The provided data URL could not be decoded.") from exc
        else:
            text = payload
        return _article_text_from_response(text, header.split(";", 1)[0].replace("data:", ""))

    parsed = urlparse(raw_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("Please enter a valid URL.")

    req = Request(raw_url, headers={"User-Agent": "TrustLens/1.0"})
    with urlopen(req, timeout=15) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        body = response.read()
        text = body.decode(charset, errors="replace")
        content_type = response.headers.get_content_type()
        return _article_text_from_response(text, content_type)


def _read_uploaded_text(file_storage):
    if file_storage is None:
        raise ValueError("Please choose a file to upload.")

    raw = file_storage.read()
    if not raw:
        raise ValueError("The uploaded file is empty.")

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="replace")

    if not text.strip():
        raise ValueError("The uploaded file does not contain readable text.")
    return text


@app.route("/")
def index():
    if _current_user():
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    records = fetch_all_analyses()
    summary = build_dashboard_summary_v2(records)
    return render_template("dashboard.html", dashboard_data=summary)


@app.route("/analyze", methods=["GET", "POST"])
@login_required
def analyze():
    article_text = ""
    source_type = request.form.get("source_type") or "text"
    url_value = ""
    analysis = None
    error = None

    if request.method == "POST":
        if request.is_json:
            payload = request.get_json(silent=True) or {}
            source_type = payload.get("source_type") or "text"
            article_text = payload.get("article_text") or payload.get("text") or ""
            url_value = payload.get("url") or ""
        else:
            article_text = request.form.get("article_text") or request.form.get("text") or ""
            url_value = request.form.get("url") or ""
            source_type = request.form.get("source_type") or "text"

        try:
            if source_type == "url":
                article_text = _read_url_text(url_value)
            elif source_type == "file":
                article_text = _read_uploaded_text(request.files.get("source_file"))
            else:
                article_text = article_text or ""

            article_text = clean_text(article_text)
            if not article_text:
                error = "Article text is required."
            else:
                analysis = analyze_news(article_text)
                analysis = _save_analysis_record(article_text, None, analysis)
        except ValueError as exc:
            error = str(exc)

        if request.is_json:
            return jsonify(_json_response(success=error is None, data=analysis, error=error))

    return render_template(
        "analyze.html",
        article_text=article_text,
        analysis=analysis,
        error=error,
        source_type=source_type,
        url_value=url_value,
    )


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
    stats = build_statistics_payload_v2(records)
    return render_template("statistics.html", stats=stats)


@app.route("/trends")
@login_required
def trends():
    records = fetch_all_analyses()
    if not records:
        return render_template("trends.html", trend_data=None)
    trend_data = build_trends_payload_v2(records)
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


@app.route("/article/<int:analysis_id>")
@login_required
def article_report(analysis_id):
    """
    Displays detailed Article Intelligence Report for a specific analysis.
    """
    records = fetch_all_analyses()
    analysis_record = None
    for r in records:
        if r.get("id") == analysis_id:
            analysis_record = r
            break
    
    if not analysis_record:
        return redirect(url_for("history"))
    
    # Generate intelligent summaries
    intelligence_summary = generate_intelligence_summary(analysis_record, analysis_record.get("article_text"))
    why_flagged = generate_why_flagged_summary(analysis_record)
    recommended_action = generate_recommended_action(analysis_record.get("risk_level", "LOW"))
    trust_interpretation = get_trust_score_interpretation(analysis_record.get("trust_score", 0))
    
    report_data = {
        "analysis": analysis_record,
        "intelligence_summary": intelligence_summary,
        "why_flagged": why_flagged,
        "recommended_action": recommended_action,
        "trust_interpretation": trust_interpretation,
        "timestamp_formatted": format_date_short(analysis_record.get("timestamp", "")),
    }
    
    return render_template("article_report.html", report=report_data)


@app.route("/compare")
@login_required
def compare():
    """
    Article comparison interface.
    Shows the comparison workflow and interface for selecting articles.
    """
    records = fetch_all_analyses()
    
    if not records or len(records) < 2:
        return render_template("compare.html", compare_data={"empty_state": True, "articles": []})
    
    return render_template("compare.html", compare_data={
        "empty_state": False,
        "articles": records[:20],  # Show recent 20 for selection
        "total_available": len(records),
    })


@app.route("/api/compare", methods=["POST"])
@login_required
def api_compare():
    """
    API endpoint for comparing two articles.
    """
    payload = request.get_json(silent=True) or {}
    try:
        article_id_1 = int(payload.get("article_id_1", 0))
        article_id_2 = int(payload.get("article_id_2", 0))
    except (ValueError, TypeError):
        return jsonify({"success": False, "error": "Invalid article IDs"})
    
    if not article_id_1 or not article_id_2:
        return jsonify({"success": False, "error": "Two article IDs required"})
    
    records = fetch_all_analyses()
    article1 = None
    article2 = None
    
    for r in records:
        if r.get("id") == article_id_1:
            article1 = r
        if r.get("id") == article_id_2:
            article2 = r
    
    if not article1 or not article2:
        return jsonify({"success": False, "error": "One or both articles not found"})
    
    comparison = {
        "article_1": {
            "id": article1.get("id"),
            "headline": article1.get("headline", "")[:100],
            "topic": article1.get("topic", "General"),
            "trust_score": article1.get("trust_score", 0),
            "bias_score": article1.get("bias_score", 0),
            "ai_probability": article1.get("ai_probability", 0),
            "headline_score": article1.get("headline_score", 50),
            "risk_level": article1.get("risk_level", "LOW"),
            "timestamp": format_date_short(article1.get("timestamp", "")),
        },
        "article_2": {
            "id": article2.get("id"),
            "headline": article2.get("headline", "")[:100],
            "topic": article2.get("topic", "General"),
            "trust_score": article2.get("trust_score", 0),
            "bias_score": article2.get("bias_score", 0),
            "ai_probability": article2.get("ai_probability", 0),
            "headline_score": article2.get("headline_score", 50),
            "risk_level": article2.get("risk_level", "LOW"),
            "timestamp": format_date_short(article2.get("timestamp", "")),
        },
        "summary": _generate_comparison_summary(article1, article2),
    }
    
    return jsonify({"success": True, "data": comparison})


def _generate_comparison_summary(article1: dict, article2: dict) -> str:
    """Generates a summary comparing two articles."""
    trust_diff = article1.get("trust_score", 0) - article2.get("trust_score", 0)
    bias_diff = article1.get("bias_score", 0) - article2.get("bias_score", 0)
    ai_diff = article1.get("ai_probability", 0) - article2.get("ai_probability", 0)
    
    parts = []
    
    if trust_diff > 10:
        parts.append("Article A has significantly higher credibility signals")
    elif trust_diff < -10:
        parts.append("Article B has significantly higher credibility signals")
    elif trust_diff > 0:
        parts.append("Article A has slightly higher credibility")
    elif trust_diff < 0:
        parts.append("Article B has slightly higher credibility")
    
    if bias_diff > 20:
        parts.append("and notably lower bias")
    elif bias_diff < -20:
        parts.append("and notably higher bias")
    
    if ai_diff > 20:
        parts.append("with lower AI-writing indicators")
    elif ai_diff < -20:
        parts.append("with higher AI-writing indicators")
    
    if not parts:
        return "The two articles show similar signal profiles."
    
    summary = ". ".join(parts) + "."
    return summary


if __name__ == "__main__":
    app.run(debug=True)
