from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

docs = []
labels = []

n = int(input("Enter number of documents: "))

for i in range(n):
    docs.append(input(f"Enter document {i+1}: "))
    labels.append(input("Enter category (contract/judgment/agreement): ").lower())

# Rule-Based Classifier
rule_pred = []

for doc in docs:

    doc = doc.lower()

    if "contract" in doc:
        rule_pred.append("contract")

    elif "judgment" in doc:
        rule_pred.append("judgment")

    else:
        rule_pred.append("agreement")

rule_acc = accuracy_score(labels, rule_pred)

# Maximum Entropy (Logistic Regression)

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(docs)

model = LogisticRegression(max_iter=1000)

model.fit(X, labels)

ml_pred = model.predict(X)

ml_acc = accuracy_score(labels, ml_pred)

print("\n========== RESULTS ==========")

print("Rule-Based Predictions:")
print(rule_pred)

print("\nMaximum Entropy Predictions:")
print(ml_pred)

print("\nRule-Based Accuracy :", round(rule_acc*100,2), "%")

print("Maximum Entropy Accuracy :", round(ml_acc*100,2), "%")
