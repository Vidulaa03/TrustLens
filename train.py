import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("dataset/news.csv")
print(data['label'].value_counts())


X = data['title'].fillna('') + " " + data['text'].fillna('')
y = data['label']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train = tfidf.fit_transform(X_train)
X_test = tfidf.transform(X_test)

# Model
model = LogisticRegression(class_weight="balanced", max_iter=1000)
model.fit(X_train, y_train)

# Accuracy
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
print("Accuracy:", acc)

# Save model
pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(tfidf, open("model/tfidf.pkl", "wb"))

print("Model saved!")
