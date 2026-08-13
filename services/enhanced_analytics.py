"""
Enhanced Analytics Module

Centralized analytics and risk calculation for TrustLens.
All metrics must derive from a single source of truth per analysis record.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

from services.topic_service import TOPIC_KEYWORDS

# ============================================================================
# RISK CALCULATION THRESHOLDS (Centralized Constants)
# ============================================================================

RISK_THRESHOLDS = {
    "trust_score_high_min": 70,      # trust_score >= 70 → LOW risk
    "trust_score_medium_min": 40,    # 40 <= trust_score < 70 → MEDIUM risk
    "bias_score_high_threshold": 65,
    "clickbait_score_high_threshold": 65,
    "headline_score_low_threshold": 40,
    "ai_probability_high_threshold": 75,
    "ai_probability_medium_threshold": 55,
}

TRUST_SCORE_COMPONENTS = {
    "credibility_weight": 0.60,          # TF-IDF + Logistic Regression model
    "emotional_language_weight": 0.10,   # Emotional/sensational signals
    "clickbait_weight": 0.10,            # Clickbait detection
    "bias_weight": 0.10,                 # Bias signals
    "headline_consistency_weight": 0.10, # Headline consistency
}

TRUST_SCORE_LABELS = {
    80: "High Confidence",
    60: "Moderate Confidence",
    40: "Low Confidence",
    0: "Very Low Confidence",
}

# ============================================================================
# RISK LEVEL CALCULATION (Centralized Function)
# ============================================================================


def calculate_risk_level(trust_score: float, bias_score: float, clickbait_score: float,
                         headline_score: float, ai_probability: float) -> str:
    """
    Calculates risk level based on measurable thresholds.
    
    HIGH RISK requires corroborating severe signals. A single heuristic is not
    enough to label an article high risk: those signals are imperfect and are
    better surfaced as MEDIUM risk for review.
    
    MEDIUM RISK:
    - trust_score < 70 (moderate credibility concerns)
    - OR bias_score > 45 (moderate bias)
    - OR clickbait_score > 35 (moderate clickbait)
    - OR headline_score < 65 (moderate headline mismatch)
    - OR ai_probability > 55 (moderate AI signals)
    
    LOW RISK:
    - trust_score >= 70
    - AND bias_score <= 45
    - AND clickbait_score <= 35
    - AND headline_score >= 65
    - AND ai_probability <= 55
    """
    severe_signals = sum((
        trust_score < 15,
        bias_score > RISK_THRESHOLDS["bias_score_high_threshold"],
        clickbait_score > RISK_THRESHOLDS["clickbait_score_high_threshold"],
        headline_score < RISK_THRESHOLDS["headline_score_low_threshold"],
        ai_probability > RISK_THRESHOLDS["ai_probability_high_threshold"],
    ))
    moderate_risk = (
        trust_score < RISK_THRESHOLDS["trust_score_high_min"]
        or bias_score > (RISK_THRESHOLDS["bias_score_high_threshold"] - 20)
        or clickbait_score > (RISK_THRESHOLDS["clickbait_score_high_threshold"] - 30)
        or headline_score < (RISK_THRESHOLDS["headline_score_low_threshold"] + 25)
        or ai_probability > RISK_THRESHOLDS["ai_probability_medium_threshold"]
    )
    
    if severe_signals >= 2:
        return "HIGH"
    if moderate_risk:
        return "MEDIUM"
    return "LOW"


def get_trust_score_interpretation(trust_score: float) -> str:
    """Returns a human-readable interpretation of trust score."""
    if trust_score >= 80:
        return TRUST_SCORE_LABELS[80]
    elif trust_score >= 60:
        return TRUST_SCORE_LABELS[60]
    elif trust_score >= 40:
        return TRUST_SCORE_LABELS[40]
    else:
        return TRUST_SCORE_LABELS[0]


# ============================================================================
# RISK DISTRIBUTION CALCULATION
# ============================================================================


def calculate_risk_distribution(records: list[dict]) -> dict[str, float]:
    """
    Calculates correct risk distribution percentages that sum to 100%.
    
    Returns:
        {
            "low_count": int,
            "medium_count": int,
            "high_count": int,
            "low_percentage": float,
            "medium_percentage": float,
            "high_percentage": float,
            "total_percentage": float,  # Should be 100.0 (or very close)
        }
    """
    if not records:
        return {
            "low_count": 0,
            "medium_count": 0,
            "high_count": 0,
            "low_percentage": 0.0,
            "medium_percentage": 0.0,
            "high_percentage": 0.0,
            "total_percentage": 0.0,
        }

    total = len(records)
    low_count = sum(1 for r in records if r.get("risk_level") == "LOW")
    medium_count = sum(1 for r in records if r.get("risk_level") == "MEDIUM")
    high_count = sum(1 for r in records if r.get("risk_level") == "HIGH")

    # Ensure counts add up to total (in case of unknown risk levels)
    other_count = total - low_count - medium_count - high_count
    if other_count > 0:
        # Redistribute unknown to LOW (conservative)
        low_count += other_count

    low_percentage = round((low_count / total) * 100, 1) if total else 0
    medium_percentage = round((medium_count / total) * 100, 1) if total else 0
    high_percentage = round((high_count / total) * 100, 1) if total else 0

    # Ensure percentages sum to 100 (handle floating point errors)
    total_percentage = low_percentage + medium_percentage + high_percentage
    if total_percentage != 100.0:
        # Adjust highest percentage to make total = 100
        if high_count >= medium_count and high_count >= low_count:
            high_percentage = round(100 - low_percentage - medium_percentage, 1)
        elif medium_count >= low_count:
            medium_percentage = round(100 - low_percentage - high_percentage, 1)
        else:
            low_percentage = round(100 - medium_percentage - high_percentage, 1)

    return {
        "low_count": low_count,
        "medium_count": medium_count,
        "high_count": high_count,
        "low_percentage": low_percentage,
        "medium_percentage": medium_percentage,
        "high_percentage": high_percentage,
        "total_percentage": round(low_percentage + medium_percentage + high_percentage, 1),
    }


# ============================================================================
# DATE FORMATTING
# ============================================================================


def format_date_short(timestamp_str: str | datetime) -> str:
    """
    Formats date for dashboard display (compact).
    
    Input: "2026-08-12 20:15:44" or datetime object
    Output: "Aug 12"
    """
    if isinstance(timestamp_str, str):
        try:
            dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return str(timestamp_str)[:10]
    else:
        dt = timestamp_str

    return dt.strftime("%b %d")


def format_date_with_time(timestamp_str: str | datetime) -> str:
    """
    Formats date for tooltip/detailed display.
    
    Input: "2026-08-12 20:15:44" or datetime object
    Output: "Aug 12 · 8 PM"
    """
    if isinstance(timestamp_str, str):
        try:
            dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return str(timestamp_str)
    else:
        dt = timestamp_str

    hour = dt.hour
    ampm = "AM" if hour < 12 else "PM"
    hour_12 = hour if hour <= 12 else hour - 12
    if hour_12 == 0:
        hour_12 = 12

    return dt.strftime("%b %d") + f" · {hour_12} {ampm}"


def format_date_iso(timestamp_str: str | datetime) -> str:
    """
    Formats date in ISO-like format.
    
    Input: "2026-08-12 20:15:44" or datetime object
    Output: "2026-08-12"
    """
    if isinstance(timestamp_str, str):
        return timestamp_str[:10]
    else:
        return timestamp_str.strftime("%Y-%m-%d")


# ============================================================================
# DASHBOARD SUMMARY (Enhanced)
# ============================================================================


def build_dashboard_summary_v2(records: list[dict]) -> dict[str, Any]:
    """
    Builds enhanced dashboard summary with:
    - Correct risk distribution percentages
    - Improved trust score interpretation
    - Proper date formatting
    - Empty state handling
    """
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
            "risk_distribution": {
                "low_count": 0,
                "medium_count": 0,
                "high_count": 0,
                "low_percentage": 0.0,
                "medium_percentage": 0.0,
                "high_percentage": 0.0,
            },
            "empty_state": True,
        }

    total = len(records)
    average_trust = round(sum(r.get("trust_score", 0) for r in records) / total, 1)
    average_bias = round(sum(r.get("bias_score", 0) for r in records) / total, 1)
    average_ai = round(sum(r.get("ai_probability", 0) for r in records) / total, 1)
    high_risk_count = sum(1 for r in records if r.get("risk_level") == "HIGH")
    topic_counts = Counter(r.get("topic") or "General" for r in records)
    
    risk_dist = calculate_risk_distribution(records)

    trend_points = []
    for record in records:
        timestamp_str = record.get("timestamp", "")
        trend_points.append({
            "date": format_date_short(timestamp_str),
            "date_full": format_date_with_time(timestamp_str),
            "trust_score": record.get("trust_score", 0),
            "bias_score": record.get("bias_score", 0),
            "ai_probability": record.get("ai_probability", 0),
            "risk_level": record.get("risk_level", "LOW"),
            "topic": record.get("topic", "General"),
        })

    return {
        "articles_analyzed": total,
        "average_trust_score": average_trust,
        # Keep the template/API aliases stable while the newer names remain
        # available to other views.
        "avg_trust_score": average_trust,
        "average_trust_interpretation": get_trust_score_interpretation(average_trust),
        "average_bias": average_bias,
        "average_ai_probability": average_ai,
        "high_risk_count": high_risk_count,
        "topics": dict(topic_counts),
        "trend_points": trend_points,
        "recent_activity": trend_points[-8:][::-1],
        "risk_distribution": risk_dist,
        "empty_state": False,
    }


# ============================================================================
# STATISTICS PAYLOAD (Enhanced)
# ============================================================================


def build_statistics_payload_v2(records: list[dict]) -> dict[str, Any] | None:
    """
    Builds enhanced statistics with consistent data from stored records.
    """
    if not records:
        return None

    total = len(records)
    average_trust = round(sum(r.get("trust_score", 0) for r in records) / total, 1)
    average_bias = round(sum(r.get("bias_score", 0) for r in records) / total, 1)
    average_ai = round(sum(r.get("ai_probability", 0) for r in records) / total, 1)
    topic_counts = Counter(r.get("topic") or "General" for r in records)
    most_common_topic = topic_counts.most_common(1)[0][0] if topic_counts else "General"
    
    risk_dist = calculate_risk_distribution(records)
    
    trust_history = [r.get("trust_score", 0) for r in records]
    bias_history = [r.get("bias_score", 0) for r in records]
    ai_history = [r.get("ai_probability", 0) for r in records]
    dates = [format_date_short(r.get("timestamp", "")) for r in records]
    peak_day = Counter(dates).most_common(1)[0][0] if dates else "N/A"

    recent_activity = []
    for row in reversed(records[-8:]):
        date_label = format_date_short(row.get("timestamp", ""))
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
            "risk_level": row.get("risk_level", "LOW"),
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
        "average_trust_score": average_trust,
        "avg_trust_score": average_trust,
        "average_trust_interpretation": get_trust_score_interpretation(average_trust),
        "avg_bias_score": average_bias,
        "avg_ai_score": average_ai,
        "most_common_topic": most_common_topic,
        "top_topic": most_common_topic,
        "peak_day": peak_day,
        "risk_distribution": risk_dist,
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


# ============================================================================
# TRENDS PAYLOAD (Enhanced)
# ============================================================================


def build_trends_payload_v2(records: list[dict]) -> dict[str, Any] | None:
    """
    Builds enhanced trends with proper empty states and data consistency.
    """
    if not records:
        return {
            "empty_state": True,
            "topic_labels": [],
            "topic_counts": [],
            "suspicious_headlines": [],
            "heatmap": [],
            "ai_dates": [],
            "ai_values": [],
        }

    topic_counts = Counter(r.get("topic") or "General" for r in records)
    trending = topic_counts.most_common(6)
    
    bias_by_topic = defaultdict(list)
    for r in records:
        bias_by_topic[r.get("topic") or "General"].append(r.get("bias_score", 0))
    avg_bias_by_topic = {topic: round(sum(values) / len(values), 1) for topic, values in bias_by_topic.items()}

    ai_dates = [format_date_short(r.get("timestamp", "")) for r in records[-10:]]
    ai_values = [r.get("ai_probability", 0) for r in records[-10:]]
    
    # Activity heatmap with proper empty state
    heatmap = []
    has_activity = False
    for day in range(7):
        row = [0 for _ in range(24)]
        for r in records:
            ts_str = r.get("timestamp", "")
            try:
                if isinstance(ts_str, str):
                    dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                else:
                    dt = ts_str
                if dt.weekday() == day:
                    row[dt.hour] += 1
                    has_activity = True
            except (ValueError, AttributeError):
                pass
        heatmap.append(row)

    # Suspicious headlines (HIGH RISK ONLY)
    suspicious = []
    high_risk_records = [r for r in records if r.get("risk_level") == "HIGH"]
    for row in sorted(high_risk_records, key=lambda x: x.get("trust_score", 0), reverse=True)[:5]:
        headline = row.get("headline") or (row.get("article_text") or "")[:80]
        signals = []
        if row.get("bias_score", 0) > 65:
            signals.append("High bias")
        if row.get("ai_probability", 0) > 75:
            signals.append("AI-assisted writing")
        if row.get("headline_score", 50) < 40:
            signals.append("Low headline consistency")
        suspicious.append({
            "headline": headline[:100],
            "risk_score": int(max(0, min(100, 100 - row.get("trust_score", 0)))),
            "risk_level": row.get("risk_level", "HIGH"),
            "signals": signals,
        })

    return {
        "empty_state": not has_activity and not suspicious,
        "topic_labels": [topic for topic, _ in trending],
        "topic_counts": [count for _, count in trending],
        "bias_cat_labels": list(avg_bias_by_topic.keys()),
        "bias_cat_values": list(avg_bias_by_topic.values()),
        "ai_dates": ai_dates,
        "ai_values": ai_values,
        "heatmap": heatmap,
        "suspicious_headlines": suspicious,
        "trending_topics": trending,
        "bias_by_topic": avg_bias_by_topic,
        "risk_signals": Counter(r.get("risk_level", "LOW") for r in records),
    }


# ============================================================================
# ARTICLE INTELLIGENCE SIGNALS
# ============================================================================


def generate_intelligence_summary(analysis: dict[str, Any], article_text: str | None = None) -> str:
    """
    Generates a concise, data-driven intelligence summary.
    
    Example:
    "TrustLens detected moderate credibility signals with low emotional framing. 
     The headline is largely consistent with the article body, while the text 
     contains several linguistic patterns associated with AI-assisted writing."
    """
    trust_score = analysis.get("trust_score", 50)
    bias_score = analysis.get("bias_score", 0)
    ai_probability = analysis.get("ai_probability", 0)
    headline_score = analysis.get("headline_score", 50)

    parts = []

    # Credibility assessment
    if trust_score >= 70:
        parts.append("strong credibility signals")
    elif trust_score >= 50:
        parts.append("moderate credibility signals")
    else:
        parts.append("low credibility signals")

    # Emotional framing
    emotional = analysis.get("emotional_score", 0)
    if emotional > 60:
        parts.append("with high emotional framing")
    elif emotional > 30:
        parts.append("with moderate emotional framing")
    else:
        parts.append("with low emotional framing")

    # Headline consistency
    if headline_score >= 75:
        if len(parts) >= 2:
            parts[-1] += ". The headline is largely consistent with the article body"
    elif headline_score >= 50:
        if len(parts) >= 2:
            parts[-1] += ". The headline has moderate consistency with the article body"
    else:
        if len(parts) >= 2:
            parts[-1] += ". The headline shows notable mismatch with the article body"

    # AI signals
    if ai_probability > 70:
        parts.append(", while the text contains linguistic patterns associated with AI-assisted writing")
    elif ai_probability > 40:
        parts.append(", though some text patterns suggest AI involvement")

    summary = "TrustLens detected " + " ".join(parts)
    if not summary.endswith("."):
        summary += "."

    return summary


def generate_why_flagged_summary(analysis: dict[str, Any]) -> list[str]:
    """
    Generates concrete reasons why an article was flagged.
    
    Returns list of specific signals detected.
    """
    signals = []
    
    trust_score = analysis.get("trust_score", 50)
    if trust_score < 40:
        signals.append(f"Low trust score: {trust_score}/100")
    elif trust_score < 60:
        signals.append(f"Moderate trust score: {trust_score}/100")

    bias_score = analysis.get("bias_score", 0)
    emotional_score = analysis.get("emotional_score", 0)
    if emotional_score > 70:
        signals.append(f"{int(emotional_score/10)}-10 emotional phrases detected")
    
    sensational_score = analysis.get("sensational_score", 0)
    if sensational_score > 60:
        signals.append("Sensational phrasing detected")

    headline_score = analysis.get("headline_score", 50)
    consistency = min(100, max(0, headline_score))
    if consistency < 65:
        signals.append(f"Headline/body consistency: {int(consistency)}%")

    ai_probability = analysis.get("ai_probability", 0)
    if ai_probability > 60:
        signals.append(f"AI-writing likelihood: {int(ai_probability)}%")

    fake_prob = analysis.get("fake_probability", 0)
    if fake_prob > 60:
        signals.append(f"Credibility model suggests potential credibility concerns")

    if not signals:
        signals.append("Model assessment indicates heightened content scrutiny is recommended")

    return signals[:5]  # Return top 5 signals


def generate_recommended_action(risk_level: str) -> str:
    """
    Generates a recommended action based on risk level.
    """
    if risk_level == "HIGH":
        return "Verify claims independently before relying on this content. Consider fact-checking key assertions with authoritative sources."
    elif risk_level == "MEDIUM":
        return "Review key claims before publication or citation. Consider cross-referencing with other sources."
    else:
        return "Suitable for normal review. This content shows strong credibility signals."
