import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# Download required resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

# Get input
text = input("Enter a sentence: ")

# Tokenize
tokens = word_tokenize(text)

# POS Tagging
tagged_words = pos_tag(tokens)

print("\nTokens:")
print(tokens)

print("\nPOS Tags:")
for word, tag in tagged_words:
    print(word, "->", tag)

print("\nTotal Words:", len(tokens))
