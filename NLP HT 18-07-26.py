import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv("pos_tags.csv", nrows=1000)

print(data.head())

# -----------------------------
# Convert into Sentences
# -----------------------------
sentences = []
temp = []

for _, row in data.iterrows():

    word = row["word"]
    tag = row["tag"]

    temp.append((word, tag))

    if len(temp) == 20:
        sentences.append(temp)
        temp = []

print("Total Sentences:", len(sentences))

# -----------------------------
# Train Test Split
# -----------------------------
train_data, test_data = train_test_split(
    sentences,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Vocabulary and Tag Set
# -----------------------------
vocabulary = set()
tags = set()

for sentence in train_data:
    for word, tag in sentence:
        vocabulary.add(word.lower())
        tags.add(tag)

vocabulary = list(vocabulary)
tags = list(tags)

word_to_index = {word: i for i, word in enumerate(vocabulary)}
tag_to_index = {tag: i for i, tag in enumerate(tags)}
index_to_tag = {i: tag for tag, i in tag_to_index.items()}

V = len(vocabulary)
T = len(tags)

print("Vocabulary Size :", V)
print("POS Tags :", T)

# -----------------------------
# HMM Matrices
# -----------------------------
initial = np.ones(T)
transition = np.ones((T, T))
emission = np.ones((T, V))

# -----------------------------
# Count Frequencies
# -----------------------------
for sentence in train_data:

    # Initial Probability
    first_tag = sentence[0][1]
    initial[tag_to_index[first_tag]] += 1

    for i, (word, tag) in enumerate(sentence):

        word = word.lower()
        tag_index = tag_to_index[tag]

        # Emission Count
        if word in word_to_index:
            emission[tag_index, word_to_index[word]] += 1

        # Transition Count
        if i > 0:

            previous_tag = sentence[i - 1][1]

            transition[
                tag_to_index[previous_tag],
                tag_index
            ] += 1

# -----------------------------
# Convert Counts to Probabilities
# -----------------------------
initial /= initial.sum()

transition /= transition.sum(
    axis=1,
    keepdims=True
)

emission /= emission.sum(
    axis=1,
    keepdims=True
)

# -----------------------------
# Log Space
# -----------------------------
log_initial = np.log(initial)
log_transition = np.log(transition)
log_emission = np.log(emission)

# -----------------------------
# Vectorized Viterbi Algorithm
# -----------------------------
def viterbi(sentence):

    n = len(sentence)

    dp = np.zeros((T, n))
    backpointer = np.zeros((T, n), dtype=int)

    # First Word
    word = sentence[0].lower()

    if word in word_to_index:
        emit = log_emission[:, word_to_index[word]]
    else:
        emit = np.log(np.ones(T) * 1e-10)

    dp[:, 0] = log_initial + emit

    # Remaining Words
    for i in range(1, n):

        word = sentence[i].lower()

        if word in word_to_index:
            emit = log_emission[:, word_to_index[word]]
        else:
            emit = np.log(np.ones(T) * 1e-10)

        scores = dp[:, i - 1][:, None] + log_transition

        backpointer[:, i] = np.argmax(scores, axis=0)

        dp[:, i] = np.max(scores, axis=0) + emit

    # Backtracking
    best = np.argmax(dp[:, -1])

    result = [best]

    for i in range(n - 1, 0, -1):
        best = backpointer[best, i]
        result.append(best)

    result.reverse()

    return [index_to_tag[i] for i in result]

# -----------------------------
# Test One Sentence
# -----------------------------
sentence = "Artificial Intelligence improves healthcare systems".split()

prediction = viterbi(sentence)

print("\nPrediction")

for word, tag in zip(sentence, prediction):
    print(f"{word:15} {tag}")

# -----------------------------
# Evaluation
# -----------------------------
actual = []
predicted = []

for sentence in test_data:

    words = [w for w, t in sentence]
    true_tags = [t for w, t in sentence]

    pred_tags = viterbi(words)

    actual.extend(true_tags)

    predicted.extend(pred_tags)

print("\nAccuracy")

print(accuracy_score(actual, predicted))

print("\nClassification Report")

print(classification_report(
    actual,
    predicted,
    zero_division=0
))

# -----------------------------
# Five Unseen Sentences
# -----------------------------
test_sentences = [

    "John likes the blue house".split(),

    "She bought a new laptop yesterday".split(),

    "Artificial Intelligence improves healthcare systems".split(),

    "The children are playing football".split(),

    "Machine learning changes the world".split()

]

print("\nPredictions on Unseen Sentences\n")

for sent in test_sentences:

    tags = viterbi(sent)

    print("Sentence :", " ".join(sent))

    for w, t in zip(sent, tags):
        print(f"{w:15} {t}")

    print("-" * 40)
