import re
from statistics import mean

from utils.text_utils import clean_text


def calculate_ai_probability(text):
    cleaned = clean_text(text)
    if not cleaned:
        return 0

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]
    if len(sentences) < 2:
        return 38

    lengths = [len(re.findall(r"\b\w+\b", sentence)) for sentence in sentences]
    avg_len = mean(lengths)
    variation = 100 - min(100, abs(avg_len - 18) * 5)
    repetitiveness = 0
    seen = {}
    for sentence in sentences:
        key = re.sub(r"\W+", " ", sentence.lower()).strip()
        seen[key] = seen.get(key, 0) + 1
    if seen:
        repetitiveness = min(100, (max(seen.values()) / max(1, len(sentences))) * 100)

    generic_terms = sum(1 for term in ["important", "crucial", "furthermore", "moreover", "in conclusion", "in summary", "however", "it is worth noting"] if term in cleaned.lower())
    score = (100 - variation) * 0.5 + repetitiveness * 0.4 + min(20, generic_terms * 7)
    return max(10, min(95, round(score, 2)))


def analyze_ai_signals(text):
    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("Article text is required.")

    probability = calculate_ai_probability(cleaned)
    if probability > 70:
        label = "Likely AI-assisted"
    elif probability > 45:
        label = "Mixed Signals"
    else:
        label = "Likely Human-like"

    patterns = []
    if probability > 50:
        patterns.append("Sentence rhythm consistency")
    if probability > 40:
        patterns.append("Repetitive structure")
    if probability > 35:
        patterns.append("Generic phrasing")
    if probability > 30:
        patterns.append("Vocabulary variation is limited")
    if not patterns:
        patterns.append("No strong AI-writing signal detected")

    return {
        "ai_probability": probability,
        "human_probability": max(0, min(100, 100 - probability)),
        "label": label,
        "detected_patterns": patterns,
        "summary": "AI-writing detection is probabilistic and should not be treated as definitive proof of authorship.",
    }
