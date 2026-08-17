import os
import re
import string
import itertools
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

def load_documents(folder_path):
    documents = {}
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.txt'):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                documents[filename] = f.read()
    return documents

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    return text.strip()

def tokenize_and_normalize(text):
    tokens = nltk.word_tokenize(text)
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens if t not in STOPWORDS and len(t) > 2]
    return ' '.join(tokens)

def preprocess_documents(documents):
    processed = {}
    for name, text in documents.items():
        cleaned = clean_text(text)
        processed[name] = tokenize_and_normalize(cleaned)
    return processed

def compute_similarity_matrix(processed_docs):
    names = list(processed_docs.keys())
    corpus = [processed_docs[name] for name in names]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return names, similarity_matrix

def generate_report(names, similarity_matrix, threshold=0.5):
    rows = []
    for i, j in itertools.combinations(range(len(names)), 2):
        score = similarity_matrix[i][j]
        status = 'Possible Plagiarism' if score >= threshold else 'Low Similarity'
        rows.append({
            'Document A': names[i],
            'Document B': names[j],
            'Similarity (%)': round(score * 100, 2),
            'Status': status
        })
    report = pd.DataFrame(rows).sort_values(by='Similarity (%)', ascending=False).reset_index(drop=True)
    return report

def run_plagiarism_detector(folder_path, threshold=0.5):
    documents = load_documents(folder_path)
    if len(documents) < 2:
        raise ValueError('Need at least two documents to compare')
    processed_docs = preprocess_documents(documents)
    names, similarity_matrix = compute_similarity_matrix(processed_docs)
    report = generate_report(names, similarity_matrix, threshold)
    return report

if __name__ == '__main__':
    FOLDER_PATH = r"C:\Semester 5\NLP Skill\Class task\assignments"
    THRESHOLD = 0.5
    report = run_plagiarism_detector(FOLDER_PATH, THRESHOLD)
    print(report)
    report.to_csv('plagiarism_report.csv', index=False)
    suspicious = report[report['Status'] == 'Possible Plagiarism']
    print('\nSuspicious Pairs:')
    print(suspicious)
