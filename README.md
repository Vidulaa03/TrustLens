<div align="center">

# 🔎 TrustLens

### ML-Based News Credibility & Content Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.x-black.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](https://scikit-learn.org/)


**Analyze the signals. Investigate the story. Make the judgment.**

</div>

---

## 📖 About

TrustLens is the working implementation of the research paper **"TrustLens: Machine Learning for Automated News Credibility Scoring,"** published at the **VSEARCH VIJANAN 2026 National Level Student Research Paper Conference**. It is a Flask-based web application that analyzes news-style content using **TF-IDF + Logistic Regression** and a suite of linguistic heuristics to surface credibility, bias, sensationalism, AI-writing patterns, topic classification, and overall content risk.

> **Note:** TrustLens is a decision-support tool, not a fact-checking authority. Its scores identify patterns that may deserve further investigation — they do not prove whether an article is true or false.

---

## 📸 Demo

<div align="center">

### Dashboard — Bias, Risk & Recent Analyses
<img width="1600" height="754" alt="WhatsApp Image 2026-09-01 at 10 22 43 PM" src="https://github.com/user-attachments/assets/fa733649-9524-4436-8bec-48b8a647bb85" />


### Session Trends & Metrics Timeline
<img width="1600" height="765" alt="WhatsApp Image 2026-09-01 at 10 22 09 PM" src="https://github.com/user-attachments/assets/b94df200-d265-4480-b68e-1b88e360dd54" />


### Risk Level Progression & Trend Summary
<img width="1600" height="760" alt="WhatsApp Image 2026-09-01 at 10 24 16 PM" src="https://github.com/user-attachments/assets/1281d6ec-4d2c-45f6-802f-4c6c931e8ea3" />


### Analyze Content
<img width="1600" height="760" alt="WhatsApp Image 2026-09-01 at 10 26 45 PM" src="https://github.com/user-attachments/assets/9c2d2688-7a21-40f5-b224-6b61da4c009c" />
<img width="1916" height="907" alt="image" src="https://github.com/user-attachments/assets/ee2584fa-6a29-46b1-ba44-7450dff15b95" />



</div>

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧠 **Credibility Analysis** | TF-IDF + Logistic Regression model trained on labeled news data |
| 📊 **Trust Score** | 0–100 credibility-oriented composite score |
| ⚖️ **Bias Detection** | Flags emotional, political, and one-sided framing |
| 🚨 **Risk Detection** | Classifies content as Low / Medium / High risk |
| 🤖 **AI-Writing Likelihood** | Linguistic indicators associated with AI-assisted writing |
| 🏷️ **Topic Classification** | Politics, Finance, Technology, Health, Sports, and more |
| 📰 **Headline Consistency** | Compares headline framing against article content |
| 📋 **Article Intelligence Reports** | Consolidated, human-readable analysis output |
| 📈 **Statistics & Trends** | Aggregate insights across analyzed articles |
| 🔍 **Article Comparison** | Side-by-side comparison of multiple articles |
| 🗂️ **Analysis History** | Session-based history of past analyses |
| 🌐 **Browser-Session Storage** | History and analytics persist only for the active browser session |

---

## 🔬 How It Works

```
Article
   ↓
Text Preprocessing
   ↓
TF-IDF + Logistic Regression
   ↓
Linguistic Analysis
   ├── Sentiment
   ├── Clickbait
   ├── Bias
   ├── One-sided Language
   └── Headline Consistency
   ↓
AI-Writing + Topic Analysis
   ↓
Trust Score + Risk Level
   ↓
Intelligence Report
   ↓
Dashboard / Statistics / Trends / Comparison
```

---

## 🧠 Machine Learning

The core credibility model uses:

- **TF-IDF Vectorization** for feature extraction
- **Logistic Regression** for classification
- Training data: `dataset/news.csv`

The model produces real/fake-news probabilities, which are combined with additional linguistic signals to generate the final credibility assessment.

**Pre-trained model assets:**

```
model/model.pkl
model/tfidf.pkl
```

---

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Backend | Python, Flask, Jinja2 |
| ML / NLP | scikit-learn, pandas, VADER Sentiment |
| Session Storage | Browser `sessionStorage` |
| Frontend | HTML, CSS, JavaScript, Chart.js |
| Testing | pytest |

---

## 📁 Project Structure

```
TrustLens/
├── app.py
├── dataset/
├── model/
├── models/
├── services/
├── templates/
├── tests/
└── train.py
```

---

## 🚀 Setup

```bash
git clone <your-repository-url>
cd TrustLens

python -m venv .venv
.venv\Scripts\activate

pip install flask pandas scikit-learn vaderSentiment pytest

python app.py
```

Open your browser at:

```
http://127.0.0.1:5000
```

Navigate to **Analyze Article**, paste at least 20 words of content, and click **Analyze**. Results remain available in the current browser session only.

---

## 🧪 Run Tests

```bash
python -m pytest
```

---

## 🔄 Retrain the Model

```bash
python train.py
```

This rebuilds:

```
model/model.pkl
model/tfidf.pkl
```

using `dataset/news.csv`.

---

## ⚠️ Limitations

TrustLens does **not** independently verify facts, sources, statistics, or claims. It does not persist analysis history or user accounts in the current browser-session deployment.

Results may be affected by:

- Training-data bias
- Dataset limitations
- Domain differences
- Linguistic ambiguity
- Model uncertainty

AI-writing likelihood is probabilistic and should **not** be treated as proof of AI authorship.

TrustLens supports investigation — it does not replace human judgment or professional fact-checking.

---

## 🎓 Research Context

TrustLens is the practical, working implementation of the methodology proposed in the following peer-reviewed research paper:

> **TrustLens: Machine Learning for Automated News Credibility Scoring**
> *Vidula Kshirsagar, Amulya Gogate*
> Mentor: *Mrs. Maitreyi Joglekar*
> Department of Information Technology, Vidyalankar School of Information Technology (VSIT), Mumbai, Maharashtra
> Published in the Proceedings of the **VSEARCH VIJANAN 2026 National Level Student Research Paper Conference**
> ISBN: **978-93-5737-889-5**

**Abstract:** The rapid growth of digital media has led to an overwhelming spread of online news, making it difficult for users to distinguish between credible and misleading information. TrustLens is an automated news credibility scoring system based on machine learning techniques that examines how articles are written and the language they use, rather than attempting to independently verify facts. The system analyzes both the title and textual content of news articles using models trained on multiple publicly available fake-news and fact-checking datasets — including the Kaggle Fake News dataset, the LIAR dataset, and the MultiFC dataset — to improve generalization and reduce dataset-specific bias. The result is a score that helps users quickly gauge how much they can trust an article, supporting the broader effort to combat misinformation.

**Keywords:** Machine learning techniques, credibility scoring, logistic regression, research-oriented, fake news detection

The application translates the paper's proposed methodology into a functioning system by combining:

> **Machine Learning + NLP + Linguistic Analysis + Credibility Scoring**

---

## 🔮 Future Scope

- Multi-dataset training
- Transformer-based models
- Multilingual analysis
- Claim verification
- Source credibility analysis
- Cross-source comparison
- Improved explainability
- External fact-checking integration

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Feel free to open an issue or submit a pull request.

---
