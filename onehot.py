import spacy
import numpy as np
import matplotlib.pyplot as plt
nlp = spacy.blank("en")

vocabulary = ["quaint", "little", "village", "nestled", "rolling", "hills", "lush", "greenery", "wise", "old",
              "sage", "spent", "days", "contemplating", "mysteries", "universe", "weaving", "tales", "magic",
              "adventure", "known", "far", "wide", "uncanny", "ability", "foresee", "events", "unfolded",
              "sought", "counsel", "times", "uncertainty", "despair", "brisk", "autumn", "morning", "leaves",
              "turned", "fiery", "hues", "red", "gold", "young", "traveler", "arrived", "humble", "abode",
              "seeking", "answers", "questions", "haunted", "years", "welcomed", "warm", "smile", "invited",
              "sit", "crackling", "fire", "talked", "long", "night", "life", "love", "pursuit", "truth"]

vocab_size = len(vocabulary)
word_to_index = {word: idx for idx, word in enumerate(vocabulary)}

def one_hot_encode(word, vocab_size):
    """Generate a one-hot encoding vector for a word."""
    one_hot_vector = np.zeros(vocab_size)
    index = word_to_index.get(word.lower())
    if index is not None:
        one_hot_vector[index] = 1
    return one_hot_vector

sentence = """
In a quaint little village nestled between rolling hills and lush greenery, there lived a wise old sage who spent his days contemplating the mysteries of the universe and weaving tales of magic and adventure.
He was known far and wide for his uncanny ability to foresee events before they unfolded, and many sought his counsel in times of uncertainty and despair.
One brisk autumn morning, as the leaves turned fiery hues of red and gold, a young traveler arrived at the sage's humble abode seeking answers to questions that had haunted him for years.
The sage welcomed him with a warm smile and invited him to sit by the crackling fire, where they talked long into the night about life, love, and the pursuit of truth.
"""

doc = nlp(sentence)
word_vector_mapping = {}
for token in doc:
    word_vector_mapping[token.text] = one_hot_encode(token.text, vocab_size)
def plot_one_hot_vectors(word_vectors, vocabulary):
    fig, axs = plt.subplots(nrows=len(word_vectors), figsize=(15, 2*len(word_vectors)))
    for i, (word, vector) in enumerate(word_vectors.items()):
        axs[i].bar(np.arange(len(vocabulary)), vector, tick_label=vocabulary)
        axs[i].set_title(f"One-Hot Encoding for '{word}'")
        axs[i].set_xlabel("Words")
        axs[i].set_ylabel("One-Hot Encoding")
    plt.tight_layout()
    plt.show()

for word, vector in word_vector_mapping.items():
    print(f"One-Hot Encoding for '{word}': {vector}")
plot_one_hot_vectors(word_vector_mapping, vocabulary)
