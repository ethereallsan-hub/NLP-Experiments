import nltk
import nltk.tokenize  # type: ignore
from nltk.stem import PorterStemmer, WordNetLemmatizer
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

text = input("Enter a sentence: ") # Tokenization
tokens = nltk.tokenize.word_tokenize(text) # Stemming
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in tokens] # Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens] # Display results
print("\nOriginal Text:")
print(text)


print("\nTokens:")
print(tokens)
print("\nStemmed Words:")
print(stemmed_words)
print("\nLemmatized Words:")
print(lemmatized_words)# Simple comparison
print("\nComparison:")
print("Stemming reduces words to root forms, which may not be meaningful.")
print("Lemmatization converts words to meaningful base forms.")
