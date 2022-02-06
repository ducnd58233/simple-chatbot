import nltk
import re
import numpy as np

from nltk.stem.porter import PorterStemmer
# nltk.download('punkt')

stemmer = PorterStemmer()

def str_normalize(s):
    s = str(s).strip().lower()
    s = re.sub(' +', ' ', s)
    return s

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word)

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag

if __name__ == "__main__":
    a = "  How long does it    takes       ?     "
    a = str_normalize(a)
    print(a)
    a = tokenize(a)
    print(a)
    stemmed_words= [stem(w) for w in a]
    print(stemmed_words)
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bag = bag_of_words(sentence, words)
    print(bag)
    