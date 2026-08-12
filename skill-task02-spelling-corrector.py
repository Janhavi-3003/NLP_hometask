import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.metrics import edit_distance
nltk.download("punkt")
nltk.download("punkt_tab")
df=pd.read_csv("spelling_corpus.csv")
print("First 5 records")
print(df.head())
vocabulary=set()
for word in df["correct_word"]:
    vocabulary.add(str(word).lower())
print("\nVocabulary size:",len(vocabulary))
query=input("\nEnter your search query:")
print("\nOriginal query:")
print(query)
query=query.lower()
tokens=word_tokenize(query)
tokens=[word for word in tokens if re.match(r"^[a-zA-Z]+$", word)]
print("\nTokens:")
print(tokens)
incorrect_words=[]
for word in tokens:
    if word not in vocabulary:
        incorrect_words.append(word)
print("\nIncorrect words:")
print(incorrect_words)
corrections={}
for incorrect_word in incorrect_words:
    best_word=None
    minimum_distance=float("inf")
    for candidate in vocabulary:
        distance=edit_distance(incorrect_word,candidate)
        if distance<minimum_distance:
            minimum_distance=distance
            best_word=candidate
    corrections[incorrect_word]=best_word
print("\nSuggested Corrections:")
for incorrect,corrected in corrections.items():
    print(incorrect,"->",corrected)
corrected_tokens=[]
for word in tokens:
    if word in corrections:
        corrected_tokens.append(corrections[word])
    else:
        corrected_tokens.append(word)
corrected_query=" ".join(corrected_tokens)
print("\nOriginal Query:")
print(query)
print("\nIncorrected Words:")
print(incorrect_words)
print("\nSuggested Corrections:")
print(corrections)
print("\nFinal Corrected query:")
print(corrected_query)
