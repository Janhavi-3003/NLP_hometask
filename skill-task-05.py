import pandas as pd
import numpy as np
from collections import defaultdict, Counter

# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("english_ewt_ud_hmm_sentences_final.csv")

print("Dataset loaded successfully!")
print(df.head())


# ============================================================
# 2. PREPARE SENTENCES
# ============================================================

sentences = []

for _, row in df.iterrows():

    words = row["words"].split()
    tags = row["pos_tags"].split()

    if len(words) == len(tags) and len(words) > 0:
        sentences.append((words, tags))

print("\nNumber of sentences:", len(sentences))


# ============================================================
# 3. FIND ALL POS TAGS
# ============================================================

tags = set()

for words, pos_tags in sentences:
    tags.update(pos_tags)

tags = sorted(tags)

print("\nPOS Tags:")
print(tags)

print("\nNumber of POS Tags:", len(tags))


# ============================================================
# 4. INITIAL PROBABILITIES
# P(tag) = Number of sentences starting with tag /
#          Total number of sentences
# ============================================================

initial_counts = Counter()

for words, pos_tags in sentences:
    initial_counts[pos_tags[0]] += 1

total_sentences = len(sentences)

initial_prob = {}

# Smoothing value
SMOOTHING = 1e-6

for tag in tags:

    initial_prob[tag] = (
        initial_counts[tag] + SMOOTHING
    ) / (
        total_sentences + SMOOTHING * len(tags)
    )


print("\nInitial Probabilities:")

for tag, prob in initial_prob.items():

    print(
        tag,
        ":",
        round(prob, 6)
    )


# ============================================================
# 5. TRANSITION PROBABILITIES
# P(tag2 | tag1)
# ============================================================

transition_counts = defaultdict(Counter)

for words, pos_tags in sentences:

    for i in range(len(pos_tags) - 1):

        current_tag = pos_tags[i]
        next_tag = pos_tags[i + 1]

        transition_counts[current_tag][next_tag] += 1


transition_prob = defaultdict(dict)

for current_tag in tags:

    total = sum(
        transition_counts[current_tag].values()
    )

    # Add smoothing to every possible transition
    denominator = (
        total + SMOOTHING * len(tags)
    )

    for next_tag in tags:

        transition_prob[current_tag][next_tag] = (
            transition_counts[current_tag][next_tag]
            + SMOOTHING
        ) / denominator


print("\nTransition Probabilities:")

for current_tag in tags:

    for next_tag in tags:

        probability = transition_prob[
            current_tag
        ][next_tag]

        # Print only meaningful probabilities
        if probability > 0.0001:

            print(
                current_tag,
                "->",
                next_tag,
                ":",
                round(probability, 4)
            )


# ============================================================
# 6. EMISSION PROBABILITIES
# P(word | tag)
# ============================================================

emission_counts = defaultdict(Counter)

for words, pos_tags in sentences:

    for word, tag in zip(words, pos_tags):

        emission_counts[tag][word] += 1


emission_prob = defaultdict(dict)

for tag in tags:

    total = sum(
        emission_counts[tag].values()
    )

    for word in emission_counts[tag]:

        emission_prob[tag][word] = (
            emission_counts[tag][word] / total
        )


print("\nSample Emission Probabilities:")

for tag in tags:

    print("\n", tag)

    count = 0

    for word, probability in emission_prob[tag].items():

        print(
            word,
            ":",
            round(probability, 4)
        )

        count += 1

        if count == 5:
            break


# ============================================================
# 7. VITERBI ALGORITHM
# ============================================================

def viterbi(words):

    # --------------------------------------------------------
    # Dynamic Programming Table
    # --------------------------------------------------------

    V = [{}]

    # --------------------------------------------------------
    # Backpointer
    # --------------------------------------------------------

    backpointer = [{}]


    # ========================================================
    # FIRST WORD
    # ========================================================

    for tag in tags:

        # Get emission probability
        emission = emission_prob[tag].get(
            words[0],
            SMOOTHING
        )

        # Get initial probability
        initial = initial_prob.get(
            tag,
            SMOOTHING
        )

        # Calculate log probability
        V[0][tag] = (
            np.log(initial)
            + np.log(emission)
        )

        backpointer[0][tag] = None


    # ========================================================
    # REMAINING WORDS
    # ========================================================

    for i in range(1, len(words)):

        V.append({})
        backpointer.append({})


        for current_tag in tags:

            # ------------------------------------------------
            # Emission probability
            # ------------------------------------------------

            emission = emission_prob[current_tag].get(
                words[i],
                SMOOTHING
            )


            best_probability = -np.inf
            best_previous_tag = None


            # ------------------------------------------------
            # Check every previous POS tag
            # ------------------------------------------------

            for previous_tag in tags:

                transition = transition_prob[
                    previous_tag
                ].get(
                    current_tag,
                    SMOOTHING
                )


                # ------------------------------------------------
                # Calculate total log probability
                # ------------------------------------------------

                probability = (
                    V[i - 1][previous_tag]
                    + np.log(transition)
                    + np.log(emission)
                )


                # ------------------------------------------------
                # Find maximum probability
                # ------------------------------------------------

                if probability > best_probability:

                    best_probability = probability

                    best_previous_tag = previous_tag


            # Store best probability
            V[i][current_tag] = best_probability

            # Store previous tag
            backpointer[i][current_tag] = (
                best_previous_tag
            )


    # ========================================================
    # FIND BEST FINAL TAG
    # ========================================================

    best_final_tag = max(
        V[-1],
        key=V[-1].get
    )


    # Start best path
    best_path = [
        best_final_tag
    ]


    # ========================================================
    # BACKTRACKING
    # ========================================================

    for i in range(
        len(words) - 1,
        0,
        -1
    ):

        best_final_tag = backpointer[i][
            best_final_tag
        ]

        best_path.append(
            best_final_tag
        )


    # Reverse path
    best_path.reverse()


    return best_path


# ============================================================
# 8. TEST THE POS TAGGER
# ============================================================

test_sentence = "The small bird flew away ."

test_words = test_sentence.split()


# Run Viterbi
predicted_tags = viterbi(
    test_words
)


# ============================================================
# 9. DISPLAY RESULT
# ============================================================

print("\n========================================")
print("HMM POS TAGGING RESULT")
print("========================================")

for word, tag in zip(
    test_words,
    predicted_tags
):

    print(
        f"{word:15} -> {tag}"
    )


# ============================================================
# 10. DISPLAY COMPLETE SENTENCE
# ============================================================

print("\n========================================")
print("INPUT SENTENCE")
print("========================================")

print(test_sentence)


print("\n========================================")
print("PREDICTED POS TAGS")
print("========================================")

print(
    " ".join(predicted_tags)
)


# ============================================================
# 11. WORD + POS TAG LIST
# ============================================================

print("\n========================================")
print("WORD + POS TAG")
print("========================================")

for word, tag in zip(
    test_words,
    predicted_tags
):

    print(
        word,
        "->",
        tag
    )
