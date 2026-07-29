# ==========================================================
# Movie Review Sentiment Analysis using NLP
# ==========================================================

# Import Libraries
import pandas as pd
import re
import string
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ==========================================================
# Download NLTK Resources
# ==========================================================

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# ==========================================================
# Load Dataset
# ==========================================================

df = pd.read_csv("IMDB Dataset.csv")

print("=" * 60)
print("First 5 Records")
print("=" * 60)
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nColumns")
print(df.columns)

print("\nMissing Values")
print(df.isnull().sum())

print("\nSentiment Count")
print(df["sentiment"].value_counts())

# ==========================================================
# Data Preprocessing
# ==========================================================

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):

    # Lowercase
    text = text.lower()

    # Remove HTML Tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove Numbers
    text = re.sub(r"\d+", "", text)

    # Remove Punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenization
    words = word_tokenize(text)

    # Remove Stopwords
    words = [word for word in words if word not in stop_words]

    # Lemmatization
    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)


print("\nCleaning Reviews...")

df["clean_review"] = df["review"].apply(preprocess_text)

print("Cleaning Completed!")

# ==========================================================
# Convert Labels
# ==========================================================

df["sentiment"] = df["sentiment"].map({
    "positive": 1,
    "negative": 0
})

# ==========================================================
# TF-IDF Feature Extraction
# ==========================================================

tfidf = TfidfVectorizer(max_features=5000)

X = tfidf.fit_transform(df["clean_review"])

y = df["sentiment"]

print("\nTF-IDF Shape:", X.shape)

# ==========================================================
# Train Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================================================
# Model Training
# ==========================================================

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("\nModel Training Completed!")

# ==========================================================
# Prediction
# ==========================================================

y_pred = model.predict(X_test)

# ==========================================================
# Evaluation Metrics
# ==========================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print("Accuracy  :", round(accuracy * 100, 2), "%")
print("Precision :", round(precision * 100, 2), "%")
print("Recall    :", round(recall * 100, 2), "%")
print("F1 Score  :", round(f1 * 100, 2), "%")

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# ==========================================================
# Confusion Matrix
# ==========================================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Negative","Positive"],
    yticklabels=["Negative","Positive"]
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")

plt.show()

# ==========================================================
# Predict New Reviews
# ==========================================================

print("\n" + "=" * 60)
print("Movie Review Prediction")
print("=" * 60)

while True:

    review = input("\nEnter Movie Review (Type 'exit' to quit): ")

    if review.lower() == "exit":
        print("Program Closed.")
        break

    clean_review = preprocess_text(review)

    vector = tfidf.transform([clean_review])

    prediction = model.predict(vector)

    probability = model.predict_proba(vector)

    if prediction[0] == 1:
        print("\nPrediction : Positive Review")
        print("Confidence :", round(probability[0][1] * 100, 2), "%")
    else:
        print("\nPrediction : Negative Review")
        print("Confidence :", round(probability[0][0] * 100, 2), "%")
