import re

from utils.text_utils import clean_text

CLICKBAIT_WORDS = ["shocking", "secret", "revealed", "must", "breaking", "explosive", "urgent", "unbelievable", "bombshell", "everyone"]


def analyze_headline_consistency(headline, body):
    cleaned_headline = clean_text(headline)
    cleaned_body = clean_text(body)
    if not cleaned_headline:
        raise ValueError("Headline is required.")
    if not cleaned_body:
        raise ValueError("Article body is required.")

    headline_tokens = set(re.findall(r"\b[a-zA-Z']+\b", cleaned_headline.lower()))
    body_tokens = set(re.findall(r"\b[a-zA-Z']+\b", cleaned_body.lower()))
    overlap = len(headline_tokens & body_tokens)
    overlap_score = min(100, max(0, round((overlap / max(1, len(headline_tokens))) * 100, 2)))

    clickbait_hits = [word for word in CLICKBAIT_WORDS if word in cleaned_headline.lower()]
    clickbait_penalty = min(35, len(clickbait_hits) * 10)
    numeric_penalty = 10 if any(char.isdigit() for char in cleaned_headline) else 0
    mismatch_score = max(0, min(100, round(overlap_score - clickbait_penalty - numeric_penalty, 2)))

    if mismatch_score >= 75 and not clickbait_hits:
        label = "Accurate"
    elif mismatch_score >= 45:
        label = "Slightly Misleading"
    elif clickbait_hits:
        label = "Clickbait"
    else:
        label = "Potentially Misleading"

    reasons = []
    if overlap_score < 40:
        reasons.append("Headline and body share limited overlap in key terms.")
    if clickbait_hits:
        reasons.append("Headline includes attention-grabbing tactics that may overstate the story.")
    if numeric_penalty:
        reasons.append("Headline includes numbers or claim-like phrasing that may not be supported in the article body.")
    if not reasons:
        reasons.append("Headline is broadly aligned with the article body and does not show strong mismatch signals.")

    suggested_headline = _proposal(cleaned_headline, cleaned_body)

    return {
        "consistency_score": int(round(mismatch_score)),
        "label": label,
        "clickbait_words": clickbait_hits,
        "mismatch_reasons": reasons,
        "suggested_headline": suggested_headline,
    }


def _proposal(headline, body):
    body_tokens = re.findall(r"\b[a-zA-Z']+\b", body.lower())
    if not body_tokens:
        return headline
    keywords = [token for token in body_tokens if len(token) > 4][:6]
    if not keywords:
        return headline
    neutral = " ".join(keyword.capitalize() for keyword in keywords[:4])
    return f"{neutral} analysis examines key developments and context" if len(headline) > 40 else neutral
