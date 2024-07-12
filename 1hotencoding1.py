import spacy
import numpy as np
import matplotlib.pyplot as plt

nlp = spacy.blank("en")
vocabulary = ["dog", "cat", "orange", "apple"]
vocab_size = len(vocabulary)
word_to_index = {word: idx for idx, word in enumerate(vocabulary)}

def one_hot_encode(word, vocab_size):
    one_hot_vector = np.zeros(vocab_size)
    index = word_to_index.get(word)
    if index is not None:
        one_hot_vector[index] = 1
    return one_hot_vector

text = """
The dog chased the cat, but the cat ignored it. The dog played with an orange, and the cat ate an apple.
"""

doc = nlp(text)
word_vector_mapping = {}
for token in doc:
    word_vector_mapping[token.text] = one_hot_encode(token.text.lower(), vocab_size)
for word, vector in word_vector_mapping.items():
    print(f"One-Hot Encoding for '{word}': {vector}")
fig, axs = plt.subplots(nrows=len(word_vector_mapping), figsize=(15, 10))
for i, (word, vector) in enumerate(word_vector_mapping.items()):
    axs[i].bar(np.arange(vocab_size), vector, tick_label=vocabulary)
    axs[i].set_title(f"One-Hot Encoding for '{word}'")
    axs[i].set_xlabel("Words")
    axs[i].set_ylabel("One-Hot Encoding")

plt.tight_layout()
plt.show()
