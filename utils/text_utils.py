import re


def clean_text(text):
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def word_count(text):
    return len(re.findall(r"\b\w+\b", clean_text(text)))


def character_count(text):
    return len(clean_text(text))
