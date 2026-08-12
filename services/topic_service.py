import re
from collections import Counter

TOPIC_KEYWORDS = {
    "Politics": ["government", "election", "minister", "president", "vote", "policy", "senate", "parliament", "campaign", "law"],
    "Finance": ["stock", "market", "bank", "economy", "inflation", "revenue", "trade", "business", "currency", "investment"],
    "Technology": ["ai", "software", "app", "data", "tech", "cybersecurity", "platform", "digital", "cloud", "algorithm"],
    "Health": ["doctor", "hospital", "disease", "medicine", "health", "virus", "patient", "clinic", "wellbeing", "treatment"],
    "Sports": ["match", "team", "player", "football", "cricket", "league", "tournament", "coach", "game", "score"],
    "Entertainment": ["movie", "actor", "music", "celebrity", "film", "show", "series", "streaming", "festival", "studio"],
    "Education": ["school", "student", "teacher", "university", "education", "learning", "college", "curriculum", "classroom"],
    "Crime": ["police", "crime", "court", "trial", "arrest", "fraud", "murder", "illegal", "investigation", "offense"],
    "General": ["news", "report", "story", "article", "community", "regional", "public", "latest", "issues", "reporting"],
}


def classify_topic(text):
    cleaned = re.findall(r"\b[a-zA-Z']+\b", text.lower())
    if not cleaned:
        return {
            "primary_topic": "General",
            "secondary_topic": "General",
            "confidence": 0,
            "scores": {"General": 0},
            "keywords": []
        }

    scores = {}
    for topic, words in TOPIC_KEYWORDS.items():
        score = 0
        for word in words:
            score += cleaned.count(word)
        scores[topic] = score

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    primary_topic, primary_score = ordered[0]
    secondary_topic, secondary_score = ordered[1] if len(ordered) > 1 else (primary_topic, primary_score)
    total = sum(value for _, value in ordered)
    confidence = round((primary_score / max(1, total)) * 100, 2) if total else 0

    keywords = []
    counter = Counter(cleaned)
    for word, _ in counter.most_common(8):
        if len(word) > 3 and not word.isdigit():
            keywords.append(word)

    return {
        "primary_topic": primary_topic,
        "secondary_topic": secondary_topic,
        "confidence": confidence,
        "scores": {topic: round((score / max(1, sum(scores.values()))) * 100, 2) for topic, score in scores.items()},
        "keywords": keywords[:10],
    }
