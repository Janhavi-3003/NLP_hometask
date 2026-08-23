# ============================================================
# IMDb LSTM SENTIMENT ANALYZER
# Excel Dataset + PyTorch LSTM
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import re
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# 2. SETTINGS
# ============================================================

DATA_PATH = r"C:\Users\ALG\IMDB_Dataset.xlsx"

MAX_VOCAB_SIZE = 10000
MAX_LENGTH = 100

EMBEDDING_DIM = 64
HIDDEN_DIM = 64

BATCH_SIZE = 128
EPOCHS = 3

LEARNING_RATE = 0.001

PAD_IDX = 0
UNK_IDX = 1


# ============================================================
# 3. DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("DEVICE")
print("=" * 60)

print("Using:", DEVICE)


# ============================================================
# 4. LOAD EXCEL DATASET
# ============================================================

print("\n")
print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

df = pd.read_excel(DATA_PATH)

print("Dataset loaded successfully!")

print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 5. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "review",
    "sentiment"
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Column '{column}' not found in dataset."
        )


print("\nRequired columns found!")


# ============================================================
# 6. REMOVE MISSING VALUES
# ============================================================

print("\n")
print("=" * 60)
print("CLEANING DATASET")
print("=" * 60)

print(
    "Rows before cleaning:",
    len(df)
)


df = df.dropna(
    subset=[
        "review",
        "sentiment"
    ]
)


# Remove duplicate reviews

df = df.drop_duplicates(
    subset=[
        "review"
    ]
)


print(
    "Rows after cleaning:",
    len(df)
)


# ============================================================
# 7. TEXT CLEANING FUNCTION
# ============================================================

def clean_text(text):

    # Convert to string
    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove HTML <br>
    text = re.sub(
        r"<br\s*/?>",
        " ",
        text
    )

    # Keep only English letters
    text = re.sub(
        r"[^a-z\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# 8. CLEAN REVIEWS
# ============================================================

print("\nCleaning reviews...")

df["cleaned_review"] = df[
    "review"
].apply(
    clean_text
)

print("Text cleaning completed!")


# ============================================================
# 9. CONVERT SENTIMENT TO NUMBERS
# ============================================================

print("\n")
print("=" * 60)
print("CONVERTING LABELS")
print("=" * 60)


df["label"] = df[
    "sentiment"
].str.lower().map(
    {
        "negative": 0,
        "positive": 1
    }
)


# Remove rows with unknown labels

df = df.dropna(
    subset=[
        "label"
    ]
)


# Convert labels to integer

df["label"] = df[
    "label"
].astype(int)


print("Negative = 0")
print("Positive = 1")


print("\nLabel distribution:")

print(
    df["label"].value_counts()
)


# ============================================================
# 10. CREATE X AND Y
# ============================================================

X = df[
    "cleaned_review"
].values

y = df[
    "label"
].values


# ============================================================
# 11. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\n")
print("=" * 60)
print("DATA SPLITTING")
print("=" * 60)


# 80% Training
# 20% Temporary

X_train, X_temp, y_train, y_temp = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


# Temporary 20%
# Split into:
# 10% Validation
# 10% Testing

X_valid, X_test, y_valid, y_test = train_test_split(

    X_temp,
    y_temp,

    test_size=0.50,

    random_state=42,

    stratify=y_temp
)


print(
    "Training   :",
    len(X_train)
)

print(
    "Validation :",
    len(X_valid)
)

print(
    "Testing    :",
    len(X_test)
)


# ============================================================
# 12. BUILD VOCABULARY
# ============================================================

print("\n")
print("=" * 60)
print("BUILDING VOCABULARY")
print("=" * 60)


counter = Counter()


for text in X_train:

    words = text.split()

    counter.update(
        words
    )


# Special tokens

vocab = {

    "<PAD>": PAD_IDX,

    "<UNK>": UNK_IDX

}


# Add most common words

for i, (word, count) in enumerate(

    counter.most_common(
        MAX_VOCAB_SIZE - 2
    ),

    start=2

):

    vocab[word] = i


print(
    "Vocabulary size:",
    len(vocab)
)


# ============================================================
# 13. ENCODE TEXT
# ============================================================

def encode_text(text):

    words = text.split()


    # Convert words to numbers

    sequence = [

        vocab.get(
            word,
            UNK_IDX
        )

        for word in words

    ]


    # Truncate

    sequence = sequence[
        :MAX_LENGTH
    ]


    # Padding

    sequence += [

        PAD_IDX

    ] * (

        MAX_LENGTH - len(sequence)

    )


    return sequence


# ============================================================
# 14. ENCODE DATASETS
# ============================================================

print("\nEncoding reviews...")


X_train_encoded = np.array([

    encode_text(text)

    for text in X_train

])


X_valid_encoded = np.array([

    encode_text(text)

    for text in X_valid

])


X_test_encoded = np.array([

    encode_text(text)

    for text in X_test

])


print(
    "Training shape:",
    X_train_encoded.shape
)

print(
    "Validation shape:",
    X_valid_encoded.shape
)

print(
    "Testing shape:",
    X_test_encoded.shape
)


# ============================================================
# 15. PYTORCH DATASET
# ============================================================

class IMDbDataset(Dataset):

    def __init__(
        self,
        X,
        y
    ):

        self.X = torch.tensor(

            X,

            dtype=torch.long

        )


        self.y = torch.tensor(

            y,

            dtype=torch.float32

        )


    def __len__(self):

        return len(
            self.X
        )


    def __getitem__(
        self,
        index
    ):

        return (

            self.X[index],

            self.y[index]

        )


# ============================================================
# 16. CREATE DATASETS
# ============================================================

train_dataset = IMDbDataset(

    X_train_encoded,

    y_train

)


valid_dataset = IMDbDataset(

    X_valid_encoded,

    y_valid

)


test_dataset = IMDbDataset(

    X_test_encoded,

    y_test

)


# ============================================================
# 17. CREATE DATALOADERS
# ============================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True

)


valid_loader = DataLoader(

    valid_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False

)


test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False

)


print("\nDataLoaders created successfully!")


# ============================================================
# 18. LSTM MODEL
# ============================================================

class SentimentLSTM(nn.Module):

    def __init__(
        self,
        vocab_size
    ):

        super().__init__()


        # ----------------------------------------------------
        # Embedding Layer
        # ----------------------------------------------------

        self.embedding = nn.Embedding(

            vocab_size,

            EMBEDDING_DIM,

            padding_idx=PAD_IDX

        )


        # ----------------------------------------------------
        # LSTM Layer
        # ----------------------------------------------------

        self.lstm = nn.LSTM(

            input_size=EMBEDDING_DIM,

            hidden_size=HIDDEN_DIM,

            batch_first=True

        )


        # ----------------------------------------------------
        # Fully Connected Layer
        # ----------------------------------------------------

        self.fc = nn.Linear(

            HIDDEN_DIM,

            1

        )


    def forward(self, x):

        # Word IDs -> Embeddings

        embedded = self.embedding(
            x
        )


        # LSTM

        output, (
            hidden,
            cell
        ) = self.lstm(
            embedded
        )


        # Last hidden state

        hidden = hidden[-1]


        # Fully connected

        output = self.fc(
            hidden
        )


        return output.squeeze(1)


# ============================================================
# 19. CREATE MODEL
# ============================================================

model = SentimentLSTM(

    len(vocab)

).to(DEVICE)


# ============================================================
# 20. LOSS FUNCTION
# ============================================================

criterion = nn.BCEWithLogitsLoss()


# ============================================================
# 21. OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(

    model.parameters(),

    lr=LEARNING_RATE

)


print("\n")
print("=" * 60)
print("MODEL CREATED")
print("=" * 60)

print(model)


# ============================================================
# 22. TRAINING FUNCTION
# ============================================================

def train_model():

    model.train()


    total_loss = 0


    predictions = []

    actuals = []


    for reviews, labels in train_loader:


        # Move to GPU / CPU

        reviews = reviews.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )


        # Clear gradients

        optimizer.zero_grad()


        # Forward pass

        outputs = model(
            reviews
        )


        # Calculate loss

        loss = criterion(

            outputs,

            labels

        )


        # Backpropagation

        loss.backward()


        # Update weights

        optimizer.step()


        total_loss += loss.item()


        # Convert logits to probability

        probabilities = torch.sigmoid(

            outputs

        )


        # Convert probability to class

        predicted = (

            probabilities >= 0.5

        ).long()


        predictions.extend(

            predicted
            .detach()
            .cpu()
            .numpy()

        )


        actuals.extend(

            labels
            .detach()
            .cpu()
            .numpy()

        )


    # Average loss

    loss = (

        total_loss /

        len(train_loader)

    )


    # Accuracy

    accuracy = accuracy_score(

        actuals,

        predictions

    )


    return loss, accuracy


# ============================================================
# 23. VALIDATION FUNCTION
# ============================================================

def validate_model():

    model.eval()


    total_loss = 0


    predictions = []

    actuals = []


    with torch.no_grad():


        for reviews, labels in valid_loader:


            reviews = reviews.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )


            # Forward pass

            outputs = model(
                reviews
            )


            # Loss

            loss = criterion(

                outputs,

                labels

            )


            total_loss += loss.item()


            # Probability

            probabilities = torch.sigmoid(

                outputs

            )


            # Prediction

            predicted = (

                probabilities >= 0.5

            ).long()


            predictions.extend(

                predicted
                .cpu()
                .numpy()

            )


            actuals.extend(

                labels
                .cpu()
                .numpy()

            )


    loss = (

        total_loss /

        len(valid_loader)

    )


    accuracy = accuracy_score(

        actuals,

        predictions

    )


    return loss, accuracy


# ============================================================
# 24. TRAINING
# ============================================================

train_losses = []

validation_losses = []

train_accuracies = []

validation_accuracies = []


print("\n")
print("=" * 60)
print("TRAINING STARTED")
print("=" * 60)


for epoch in range(
    EPOCHS
):


    train_loss, train_accuracy = (

        train_model()

    )


    valid_loss, valid_accuracy = (

        validate_model()

    )


    train_losses.append(

        train_loss

    )


    validation_losses.append(

        valid_loss

    )


    train_accuracies.append(

        train_accuracy

    )


    validation_accuracies.append(

        valid_accuracy

    )


    print(

        f"Epoch {epoch + 1}/{EPOCHS} | "

        f"Train Loss: "
        f"{train_loss:.4f} | "

        f"Validation Loss: "
        f"{valid_loss:.4f} | "

        f"Train Accuracy: "
        f"{train_accuracy * 100:.2f}% | "

        f"Validation Accuracy: "
        f"{valid_accuracy * 100:.2f}%"

    )


# ============================================================
# 25. TEST MODEL
# ============================================================

print("\n")
print("=" * 60)
print("TESTING MODEL")
print("=" * 60)


model.eval()


predictions = []

actuals = []


with torch.no_grad():


    for reviews, labels in test_loader:


        reviews = reviews.to(
            DEVICE
        )


        outputs = model(
            reviews
        )


        probabilities = torch.sigmoid(

            outputs

        )


        predicted = (

            probabilities >= 0.5

        ).long()


        predictions.extend(

            predicted
            .cpu()
            .numpy()

        )


        actuals.extend(

            labels
            .numpy()

        )


# ============================================================
# 26. CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(

    actuals,

    predictions

)


precision = precision_score(

    actuals,

    predictions,

    zero_division=0

)


recall = recall_score(

    actuals,

    predictions,

    zero_division=0

)


f1 = f1_score(

    actuals,

    predictions,

    zero_division=0

)


# ============================================================
# 27. DISPLAY PERFORMANCE
# ============================================================

print("\n")
print("=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)


print(

    f"Accuracy  : "
    f"{accuracy * 100:.2f}%"

)


print(

    f"Precision : "
    f"{precision * 100:.2f}%"

)


print(

    f"Recall    : "
    f"{recall * 100:.2f}%"

)


print(

    f"F1-Score  : "
    f"{f1 * 100:.2f}%"

)


# ============================================================
# 28. LOSS VALUES
# ============================================================

print("\n")
print("=" * 60)
print("TRAINING / VALIDATION LOSS")
print("=" * 60)


for i in range(
    EPOCHS
):

    print(

        f"Epoch {i + 1}: "

        f"Training Loss = "
        f"{train_losses[i]:.4f}, "

        f"Validation Loss = "
        f"{validation_losses[i]:.4f}"

    )


# ============================================================
# 29. ACCURACY GRAPH
# ============================================================

plt.figure(

    figsize=(8, 5)

)


plt.plot(

    range(
        1,
        EPOCHS + 1
    ),

    train_accuracies,

    marker="o",

    label="Training Accuracy"

)


plt.plot(

    range(
        1,
        EPOCHS + 1
    ),

    validation_accuracies,

    marker="o",

    label="Validation Accuracy"

)


plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.title(
    "Training and Validation Accuracy"
)

plt.legend()

plt.grid()

plt.show()


# ============================================================
# 30. LOSS GRAPH
# ============================================================

plt.figure(

    figsize=(8, 5)

)


plt.plot(

    range(
        1,
        EPOCHS + 1
    ),

    train_losses,

    marker="o",

    label="Training Loss"

)


plt.plot(

    range(
        1,
        EPOCHS + 1
    ),

    validation_losses,

    marker="o",

    label="Validation Loss"

)


plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "Training and Validation Loss"
)

plt.legend()

plt.grid()

plt.show()


# ============================================================
# 31. PREDICTION FUNCTION
# ============================================================

def predict_sentiment(
    review
):


    model.eval()


    # Clean input

    cleaned = clean_text(
        review
    )


    # Convert to numbers

    sequence = encode_text(
        cleaned
    )


    # Convert to tensor

    tensor = torch.tensor(

        [sequence],

        dtype=torch.long

    ).to(DEVICE)


    # Prediction

    with torch.no_grad():


        output = model(
            tensor
        )


        probability = torch.sigmoid(

            output

        ).item()


    # Positive

    if probability >= 0.5:


        sentiment = "Positive"


        confidence = (

            probability * 100

        )


    # Negative

    else:


        sentiment = "Negative"


        confidence = (

            (1 - probability) * 100

        )


    print("\n")

    print("=" * 60)

    print(
        "INPUT:"
    )

    print(
        review
    )
    print("\n")

    print(
        "PREDICTED SENTIMENT:",
        sentiment
    )

    print(

        f"CONFIDENCE: "
        f"{confidence:.2f}%"

    )


    print("=" * 60)


# ============================================================
# 32. TEST CUSTOM REVIEW
# ============================================================

predict_sentiment(

    "The movie was excellent and very enjoyable."

)
