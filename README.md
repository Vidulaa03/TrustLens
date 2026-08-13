#  TrustLens

## ML-Based News Credibility & Content Intelligence Platform

TrustLens is the working implementation of the research paper *“TrustLens: An Automated News Credibility Scoring System Based on Machine Learning Techniques.”*

TrustLens is a Flask-based web application that analyzes news-style content using TF-IDF + Logistic Regression and linguistic heuristics to identify credibility, bias, sensationalism, AI-writing patterns, topic, and overall content risk.

> TrustLens is a decision-support tool, not a fact-checking authority. Its scores identify patterns that may deserve further investigation; they do not prove whether an article is true or false.

## Features

- 🧠 **Credibility Analysis** — TF-IDF + Logistic Regression
- 📊 **Trust Score** — 0–100 credibility-oriented score
- ⚖️ **Bias Detection** — emotional, political, and one-sided framing
- 🚨 **Risk Detection** — Low / Medium / High
- 🤖 **AI-Writing Likelihood** — linguistic indicators associated with AI-assisted writing
- 🏷️ **Topic Classification** — Politics, Finance, Technology, Health, Sports, and more
- 📰 **Headline Consistency** — compares headline and article content
- 📋 **Article Intelligence Reports**
- 📈 **Statistics & Trends**
- 🔍 **Article Comparison**
- 🗂️ **Analysis History**
- 👤 **User Authentication & Profiles**
- 💾 **SQLite Persistence**

## 🔬 How It Works

```text
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

## 🧠 Machine Learning

The core credibility model uses:

- TF-IDF Vectorization
- Logistic Regression
- `dataset/news.csv`

The model produces real/fake-news probabilities, which are combined with additional linguistic signals to generate the final assessment.

Pre-trained model assets:

```text
model/model.pkl
model/tfidf.pkl
```

## 🛠️ Tech Stack

| Category | Technologies |
| --- | --- |
| Backend | Python, Flask, Jinja2 |
| ML/NLP | scikit-learn, pandas, VADER Sentiment |
| Database | SQLite |
| Frontend | HTML, CSS, JavaScript, Chart.js |
| Testing | pytest |

## 📁 Project Structure

```text
TrustLens/
├── app.py
├── dataset/
├── model/
├── models/
├── services/
├── templates/
├── tests/
├── utils/
├── train.py
└── trustlens.db
```

## 🚀 Setup

```bash
git clone <your-repository-url>
cd TrustLens

python -m venv .venv
.venv\Scripts\activate

pip install flask pandas scikit-learn vaderSentiment pytest

python app.py
```

Open:

```text
http://127.0.0.1:5000
```

Create an account → **Analyze Article** → paste at least 20 words → analyze.

## 🧪 Run Tests

```bash
python -m pytest
```

## 🔄 Retrain the Model

```bash
python train.py
```

This rebuilds:

```text
model/model.pkl
model/tfidf.pkl
```

using `dataset/news.csv`.

## ⚠️ Limitations

TrustLens does not independently verify facts, sources, statistics, or claims.

Results may be affected by:

- Training-data bias
- Dataset limitations
- Domain differences
- Linguistic ambiguity
- Model uncertainty

AI-writing likelihood is probabilistic and should not be treated as proof of AI authorship.

TrustLens supports investigation — it does not replace human judgment or professional fact-checking.

## 🎓 Research Context

TrustLens translates the methodology proposed in the research paper into a working application by combining:

> **Machine Learning + NLP + Linguistic Analysis + Credibility Scoring**

### Research Paper

*TrustLens: An Automated News Credibility Scoring System Based on Machine Learning Techniques*

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

## 🔎 TrustLens

> **Analyze the signals. Investigate the story. Make the judgment.**
