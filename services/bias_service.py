import re

from utils.text_utils import clean_text

EMOTIONAL_WORDS = {
    "shocking", "outraged", "corrupt", "fraud", "evil", "lies", "bombshell", "devastating",
    "scandal", "panic", "horrific", "disaster", "outrage", "secret", "must", "obviously",
    "everyone", "never", "always", "ruthless", "broken", "catastrophe"
}
SENSATIONAL_PHRASES = [
    "you won't believe", "shocking", "secret", "bombshell", "revealed", "must see",
    "everyone knows", "obviously", "breaking", "hidden truth", "never before"
]
OPINION_INDICATORS = ["obviously", "clearly", "everyone", "nobody", "simply", "frankly", "unfortunately"]


def analyze_bias(text):
    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("Article text is required.")

    words = re.findall(r"\b[a-zA-Z']+\b", cleaned.lower())
    emotional_words = sorted({word for word in words if word in EMOTIONAL_WORDS})

    found_phrases = []
    text_lower = cleaned.lower()
    for phrase in SENSATIONAL_PHRASES:
        if phrase in text_lower:
            found_phrases.append(phrase)

    opinion_hits = [word for word in words if word in OPINION_INDICATORS]
    sentence_hits = []
    for sentence in re.split(r"(?<=[.!?])\s+", cleaned):
        if sentence and any(word in sentence.lower() for word in emotional_words or ["shocking", "corrupt", "obviously", "secret", "bombshell"]):
            sentence_hits.append(sentence.strip())

    score = min(100, max(0, len(emotional_words) * 12 + len(found_phrases) * 10 + len(opinion_hits) * 8 + cleaned.count("!") * 5))
    if score < 30:
        classification = "Mostly Neutral"
    elif score < 60:
        classification = "Somewhat Biased"
    else:
        classification = "Highly Biased"

    return {
        "bias_score": round(score, 2),
        "classification": classification,
        "emotional_words": emotional_words[:12],
        "sensational_phrases": found_phrases[:8],
        "opinion_indicators": sorted(set(opinion_hits))[:8],
        "highlighted_sentences": sentence_hits[:5],
        "summary": "Detected language patterns suggest the article uses emotionally charged or opinionated framing.",
    }
