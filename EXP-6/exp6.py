import nltk
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Download required resources
nltk.download('punkt')

# Biomedical relation keywords
keywords = [
    "treats", "reduces", "controls", "helps",
    "causes", "prevents", "induces", "improves",
    "inhibits", "increases", "decreases",
    "associated", "linked", "activates",
    "suppresses", "regulates"
]

# Input paragraph
text = input("Enter Biomedical Text:\n")

# Split into sentences
sentences = sent_tokenize(text)

results = []

print("\n==============================")
print("Biomedical Relation Extraction")
print("==============================")

for i, sentence in enumerate(sentences):

    tokens = word_tokenize(sentence.lower())

    predicted = 0
    relation_word = "None"

    for word in tokens:
        if word in keywords:
            predicted = 1
            relation_word = word
            break

    print(f"\nSentence {i+1}")
    print(sentence)
    print("Relation Word :", relation_word)
    print("Prediction    :", predicted)

    actual = int(input("Actual Relation (1/0): "))

    results.append([sentence, relation_word, actual, predicted])

# Create DataFrame
df = pd.DataFrame(results,
                  columns=["Sentence",
                           "Relation Word",
                           "Actual",
                           "Predicted"])

print("\nResult Table\n")
print(df)

# Evaluation Metrics
y_true = df["Actual"]
y_pred = df["Predicted"]

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print("\n========== Evaluation ==========")
print("Accuracy :", round(accuracy,3))
print("Precision:", round(precision,3))
print("Recall   :", round(recall,3))
print("F1 Score :", round(f1,3))

# Save Results
df.to_csv("Biomedical_Relation_Results.csv", index=False)
print("\nResults saved as Biomedical_Relation_Results.csv")

# Frequency of detected relation words
relation_count = df["Relation Word"].value_counts()

print("\nRelation Word Frequency\n")
print(relation_count)

# Plot graph
plt.figure(figsize=(8,5))
relation_count.plot(kind="bar")
plt.title("Detected Biomedical Relation Words")
plt.xlabel("Relation Word")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# Search sentences
while True:

    search = input("\nSearch Relation Word (EXIT to stop): ")

    if search.upper() == "EXIT":
        break

    result = df[df["Relation Word"].str.lower() == search.lower()]

    if len(result) == 0:
        print("Relation Word Not Found")
    else:
        print(result)

print("\nProgram Completed Successfully!")
