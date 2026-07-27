import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

# Input documents
docs = []
n = int(input("Enter number of documents: "))

for i in range(n):
    docs.append(input("Enter document: "))

# Input search query
query = input("\nEnter search query: ")

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)

# Query vector
query_vec = vectorizer.transform([query])

# Classical IR (TF-IDF + Cosine Similarity)
scores = cosine_similarity(query_vec, X)

print("\nTF-IDF Similarity Scores:")
for i, s in enumerate(scores[0]):
    print("Document", i + 1, ":", round(s, 3))

# Non-Classical IR (LSA using Truncated SVD)
n_components = min(2, X.shape[1] - 1) if X.shape[1] > 1 else 1
svd = TruncatedSVD(n_components=n_components, random_state=42)

X_lsa = svd.fit_transform(X)
query_lsa = svd.transform(query_vec)

lsa_scores = cosine_similarity(query_lsa, X_lsa)

print("\nLSA Similarity Scores:")
for i, s in enumerate(lsa_scores[0]):
    print("Document", i + 1, ":", round(s, 3))

# Most relevant document
best = np.argmax(lsa_scores)

print("\nMost Relevant Document:")
print(docs[best])
