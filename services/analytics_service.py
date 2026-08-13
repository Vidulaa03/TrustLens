from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

from services.topic_service import TOPIC_KEYWORDS


def classify_risk(trust_score: float, bias_score: float, clickbait_score: float, headline_score: float, ai_probability: float) -> str:
    strong_risk = (
        trust_score < 45
        or bias_score > 65
        or clickbait_score > 65
        or headline_score < 40
        or ai_probability > 75
    )
    moderate_risk = (
        trust_score < 70
        or bias_score > 45
        or clickbait_score > 35
        or headline_score < 65
        or ai_probability > 55
    )
    if strong_risk:
        return "HIGH"
    if moderate_risk:
        return "MEDIUM"
    return "LOW"


def build_dashboard_summary(records):
    if not records:
        return {
            "articles_analyzed": 0,
            "average_trust_score": 0,
            "average_bias": 0,
            "average_ai_probability": 0,
            "high_risk_count": 0,
            "topics": {},
            "trend_points": [],
            "recent_activity": [],
        }

    total = len(records)
    average_trust = round(sum(r["trust_score"] for r in records) / total, 2)
    average_bias = round(sum(r["bias_score"] for r in records) / total, 2)
    average_ai = round(sum(r["ai_probability"] for r in records) / total, 2)
    high_risk_count = sum(1 for r in records if r["risk_level"] == "HIGH")
    topic_counts = Counter(r["topic"] or "General" for r in records)

    trend_points = []
    for record in records:
        trend_points.append({
            "date": record["timestamp"].strftime("%b %d") if isinstance(record["timestamp"], datetime) else str(record["timestamp"]),
            "trust_score": record["trust_score"],
            "bias_score": record["bias_score"],
            "ai_probability": record["ai_probability"],
            "risk_level": record["risk_level"],
            "topic": record["topic"],
        })

    return {
        "articles_analyzed": total,
        "average_trust_score": average_trust,
        "average_bias": average_bias,
        "average_ai_probability": average_ai,
        "high_risk_count": high_risk_count,
        "topics": dict(topic_counts),
        "trend_points": trend_points,
        "recent_activity": trend_points[-8:][::-1],
    }


def build_statistics_payload(records):
    if not records:
        return None

    total = len(records)
    average_trust = round(sum(r["trust_score"] for r in records) / total, 2)
    average_bias = round(sum(r["bias_score"] for r in records) / total, 2)
    average_ai = round(sum(r["ai_probability"] for r in records) / total, 2)
    topic_counts = Counter(r["topic"] or "General" for r in records)
    most_common_topic = topic_counts.most_common(1)[0][0]
    high_risk_percentage = round((sum(1 for r in records if r["risk_level"] == "HIGH") / total) * 100, 2)

    trust_history = [r["trust_score"] for r in records]
    bias_history = [r["bias_score"] for r in records]
    ai_history = [r["ai_probability"] for r in records]
    dates = [r["timestamp"].strftime("%b %d") if hasattr(r["timestamp"], "strftime") else str(r["timestamp"]) for r in records]
    peak_day = Counter(dates).most_common(1)[0][0] if dates else "N/A"

    recent_activity = []
    for row in reversed(records[-8:]):
        date_label = row["timestamp"].strftime("%b %d") if hasattr(row["timestamp"], "strftime") else str(row["timestamp"])
        article_text = row.get("article_text") or ""
        snippet = article_text[:60]
        if len(article_text) > 60:
            snippet += "..."
        recent_activity.append({
            "date": date_label,
            "snippet": snippet,
            "trust": int(row.get("trust_score", 0)),
            "bias": int(row.get("bias_score", 0)),
            "topic": row.get("topic") or "General",
            "ai_pct": int(row.get("ai_probability", 0)),
        })

    chart_data = {
        "dates": dates,
        "trust_scores": trust_history,
        "bias_scores": bias_history,
        "ai_scores": ai_history,
        "topic_labels": list(topic_counts.keys()),
        "topic_counts": list(topic_counts.values()),
    }

    return {
        "total_analyses": total,
        "total_articles": total,
        "average_trust": average_trust,
        "avg_trust_score": average_trust,
        "average_bias": average_bias,
        "avg_bias_score": average_bias,
        "average_ai_probability": average_ai,
        "avg_ai_score": average_ai,
        "most_common_topic": most_common_topic,
        "top_topic": most_common_topic,
        "peak_day": peak_day,
        "high_risk_percentage": high_risk_percentage,
        "trust_history": trust_history,
        "bias_history": bias_history,
        "ai_history": ai_history,
        "dates": dates,
        "topic_counts": dict(topic_counts),
        "topic_labels": list(topic_counts.keys()),
        "topic_values": list(topic_counts.values()),
        "chart_data": chart_data,
        "recent_activity": recent_activity,
    }


def build_trends_payload(records):
    if not records:
        return None

    topic_counts = Counter(r["topic"] or "General" for r in records)
    trending = topic_counts.most_common(6)
    bias_by_topic = defaultdict(list)
    for r in records:
        bias_by_topic[r["topic"] or "General"].append(r["bias_score"])
    avg_bias_by_topic = {topic: round(sum(values) / len(values), 2) for topic, values in bias_by_topic.items()}

    ai_dates = [r["timestamp"].strftime("%b %d") if hasattr(r["timestamp"], "strftime") else str(r["timestamp"]) for r in records[-10:]]
    ai_values = [r["ai_probability"] for r in records[-10:]]
    heatmap = []
    for day in range(7):
        row = [0 for _ in range(24)]
        for r in records:
            if not hasattr(r["timestamp"], "weekday"):
                continue
            if r["timestamp"].weekday() == day:
                row[r["timestamp"].hour] = row[r["timestamp"].hour] + 1
        heatmap.append(row)

    suspicious = []
    for row in sorted(records, key=lambda item: item.get("risk_level", "LOW"), reverse=True)[:5]:
        headline = row.get("headline") or (row.get("article_text") or "")[:80]
        suspicious.append({
            "headline": headline[:80],
            "risk": int(max(0, min(100, row.get("trust_score", 0))))
        })

    anomalies = []
    for risk_level, count in Counter(r["risk_level"] for r in records).items():
        if count:
            anomalies.append({
                "severity": "high" if risk_level == "HIGH" else "medium" if risk_level == "MEDIUM" else "low",
                "title": f"{risk_level} risk signal",
                "description": f"{count} item(s) flagged as {risk_level.lower()} risk."
            })

    return {
        "topic_labels": [topic for topic, _ in trending],
        "topic_counts": [count for _, count in trending],
        "bias_cat_labels": list(avg_bias_by_topic.keys()),
        "bias_cat_values": list(avg_bias_by_topic.values()),
        "ai_dates": ai_dates,
        "ai_values": ai_values,
        "heatmap": heatmap,
        "suspicious_headlines": suspicious,
        "anomalies": anomalies,
        "trending_topics": trending,
        "bias_by_topic": avg_bias_by_topic,
        "ai_over_time": ai_values,
        "risk_signals": Counter(r["risk_level"] for r in records),
        "suspicious": [r for r in records if r["risk_level"] in {"HIGH", "MEDIUM"}],
    }
