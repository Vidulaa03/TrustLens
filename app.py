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
import os
import re
from datetime import datetime
from html.parser import HTMLParser
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from flask import Flask, jsonify, redirect, render_template, request, url_for
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


@app.route("/api/auth/register", methods=["POST"])
def api_register():
    return jsonify({"success": False, "error": "Authentication is disabled in browser-session mode."}), 410


@app.route("/api/auth/login", methods=["POST"])
def api_login():
    return jsonify({"success": False, "error": "Authentication is disabled in browser-session mode."}), 410


@app.route("/login", methods=["GET", "POST"])
def login():
    return redirect(url_for("analyze"))


@app.route("/register", methods=["GET", "POST"])
def register():
    return redirect(url_for("analyze"))


@app.route("/logout")
def logout():
    return redirect(url_for("analyze"))


def _json_response(success, data=None, error=None):
    return {"success": success, "data": data, "error": error}


def _save_analysis_record(article_text, headline, analysis):
    risk_level = calculate_risk_level(
        analysis.get("trust_score", 0),
        analysis.get("bias_score", 0),
        analysis.get("clickbait_score", 0),
        analysis.get("headline_score", 50),
        analysis.get("ai_probability", 0),
    )
    analysis["risk_level"] = risk_level
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
    return redirect(url_for("analyze"))


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/analyze", methods=["GET", "POST"])
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
def history():
    return render_template("history.html", analyses=[], topics=[], risks=[])


@app.route("/history/delete/<int:analysis_id>", methods=["POST"])
def delete_history_entry(analysis_id):
    return redirect(url_for("history"))


@app.route("/history/export.csv")
def export_history_csv():
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "date", "article", "topic", "trust_score", "bias_score", "ai_probability", "risk_level"])
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
def statistics():
    return render_template("statistics.html", stats=None)


@app.route("/trends")
def trends():
    return render_template("trends.html", trend_data=None)


@app.route("/settings", methods=["GET", "POST"])
def settings():
    return render_template("settings.html")


@app.route("/api-access")
def api_access():
    return render_template("api_access.html", api_status="Connected", usage_total=0, usage_limit=1000)


@app.route("/profile")
def profile():
    return redirect(url_for("analyze"))


@app.route("/article/<int:analysis_id>")
def article_report(analysis_id):
    """
    Displays detailed Article Intelligence Report for a specific analysis.
    """
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
def compare():
    """
    Article comparison interface.
    Shows the comparison workflow and interface for selecting articles.
    """
    return render_template("compare.html", compare_data={"empty_state": True, "articles": []})


@app.route("/api/compare", methods=["POST"])
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
    
    records = payload.get("articles") or []
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
    app.run(
        host="127.0.0.1",
        port=int(os.environ.get("PORT", 5000)),
        debug=os.environ.get("FLASK_DEBUG") == "1",
    )
