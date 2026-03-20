# models/predictor.py

from difflib import get_close_matches

class Predictor:
    def __init__(self, corpus_text):
        words = corpus_text.lower().split()
        self.words = list(set(words))
        self.bigrams = {}
        for i in range(len(words)-1):
            w1, w2 = words[i], words[i+1]
            self.bigrams.setdefault(w1, []).append(w2)

    def suggest(self, last_word):
        return self.bigrams.get(last_word.lower(), [])[:3]

    def correct(self, word):
        match = get_close_matches(word.lower(), self.words, n=1, cutoff=0.7)
        return match[0] if match else word
