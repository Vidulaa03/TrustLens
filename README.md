#  TrustLens AI Content Intelligence Platform

TrustLens is an AI-powered multi-dimensional content intelligence system designed to evaluate news credibility using machine learning, emotional intensity modeling, and explainable AI techniques.

It combines traditional NLP models with credibility scoring logic to assess whether a piece of content is likely real or misleading.

---

##  Core Capabilities

- 🎯 Credibility Classification (TF-IDF + Logistic Regression)
- 🔥 Emotional Intensity Detection (VADER Sentiment)
- 💥 Clickbait Detection
- 🔎 Explainable AI (Influential Word Highlighting)
- 📊 Multi-dimensional Trust Score Calculation
- 🎨 Futuristic Interactive Dashboard UI
- 🌌 Animated Particle Background (Live UI)
- 🧩 Modular Architecture for Future Expansion

---

##  Model Details

The credibility engine is built using:

- TF-IDF Vectorization (max_features=5000)
- Logistic Regression (balanced class weights)
- Multi-dataset training (News + LIAR dataset)
- 80/20 Train-Test Split

Performance Metrics (Baseline):
- Accuracy: ~93%
- Precision: High confidence classification
- Recall: Balanced detection across fake/real

---

##  Tech Stack

**Backend**
- Python
- Flask
- scikit-learn
- pandas
- NumPy
- VADER Sentiment

**Frontend**
- HTML5
- CSS3 (Glassmorphism + Gradients)
- JavaScript (Canvas-based particle animation)

**Version Control**
- Git
- GitHub (feature-branch workflow)

---

## 📁 Project Structure

```
TrustLens/
│
├── dataset/
│   ├── news.csv
│   ├── train.tsv
│
├── templates/
│   ├── base.html
│   ├── dashboard.html
│   ├── analyze.html
│
├── app.py
├── requirements.txt
└── README.md
```

---

##  Installation & Setup

1. Clone the repository:

```
git clone https://github.com/your-username/TrustLens.git
cd TrustLens
```

2. Create virtual environment:

```
python -m venv .venv
```

3. Activate environment:

Windows:
```
.venv\Scripts\activate
```

Mac/Linux:
```
source .venv/bin/activate
```

4. Install dependencies:

```
pip install -r requirements.txt
```

5. Run the app:

```
python app.py
```

Then visit:

```
http://127.0.0.1:5000/
```

---

##  Planned Enhancements (v2+ Roadmap)

- ⚖ Bias Detection Module
- 🧠 Topic Classification
- 🔎 Headline-Body Consistency Analysis
- 🤖 AI-Generated Content Detection
- 📈 Historical Trend Tracking
- 🗄 Database Integration (SQLite/PostgreSQL)
- 📊 Analytics Dashboard
- 🌐 REST API Endpoints

---

##  Purpose of This Project

TrustLens was built as a full-stack AI application to explore:

- Applied Machine Learning in real-world misinformation detection
- Explainable AI techniques
- Scalable web application architecture
- Interactive UI/UX design for ML systems

---
⭐ If you found this project interesting, feel free to star the repository!
