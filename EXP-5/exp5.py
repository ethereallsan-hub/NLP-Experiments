import nltk
import pandas as pd
import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag, ne_chunk
from nltk.tree import Tree

# ==========================
# Download Required Resources
# ==========================

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')

# ==========================
# Input Legal Text
# ==========================

text = input("Enter Legal Text:\n")

# ==========================
# Sentence Tokenization
# ==========================

sentences = sent_tokenize(text)

# ==========================
# Word Tokenization
# ==========================

tokens = word_tokenize(text)

# ==========================
# POS Tagging
# ==========================

tags = pos_tag(tokens)

# ==========================
# Named Entity Recognition
# ==========================

tree = ne_chunk(tags)

print("\nDetected Named Entities\n")

entities = []

for subtree in tree:

    if isinstance(subtree, Tree):

        entity = " ".join(word for word, tag in subtree.leaves())
        entity_type = subtree.label()

        print(entity, "->", entity_type)

        entities.append([entity, entity_type])

# ==========================
# Create DataFrame
# ==========================

df = pd.DataFrame(entities, columns=["Entity", "Type"])

print("\nEntity Table\n")

print(df)

# ==========================
# Entity Counts
# ==========================

if len(df) > 0:

    entity_counts = df["Type"].value_counts()

    print("\nEntity Counts\n")
    print(entity_counts)

    # ==========================
    # Graph
    # ==========================

    plt.figure(figsize=(7,5))

    entity_counts.plot(kind="bar")

    plt.title("Named Entity Distribution")
    plt.xlabel("Entity Type")
    plt.ylabel("Count")

    plt.grid(True)

    plt.show()

else:

    print("\nNo Named Entities Detected.")

# ==========================
# Save CSV
# ==========================

df.to_csv("Detected_Entities.csv", index=False)

print("\nDetected entities saved as Detected_Entities.csv")

# ==========================
# Evaluation
# ==========================

predicted = len(df)

print("\nPredicted Entities :", predicted)

actual = int(input("Enter Actual Number of Entities: "))

correct = int(input("Enter Correctly Detected Entities: "))

precision = correct / predicted if predicted > 0 else 0
recall = correct / actual if actual > 0 else 0

if precision + recall == 0:
    f1 = 0
else:
    f1 = (2 * precision * recall) / (precision + recall)

print("\n========== Evaluation ==========")
print("Sentences          :", len(sentences))
print("Words              :", len(tokens))
print("Predicted Entities :", predicted)
print("Precision          :", round(precision, 3))
print("Recall             :", round(recall, 3))
print("F1 Score           :", round(f1, 3))

# ==========================
# Search Entity
# ==========================

while True:

    search = input("\nSearch Entity (Type EXIT to stop): ")

    if search.upper() == "EXIT":
        break

    result = df[df["Entity"].str.lower() == search.lower()]

    if len(result) == 0:
        print("Entity Not Found")
    else:
        print(result)

print("\nProgram Completed Successfully!")
