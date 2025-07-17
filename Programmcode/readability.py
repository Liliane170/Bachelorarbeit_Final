from transformers import AutoTokenizer
import re


class Readability():
 
    
    def count_syllables(self, word):
        word = word.lower()
        vowels = "aeiouy"
        count = 0
        prev_char_was_vowel = False

        for char in word:
            if char in vowels:
                if not prev_char_was_vowel:
                    count += 1
                    prev_char_was_vowel = True
            else:
                prev_char_was_vowel = False

        if word.endswith("e") and count > 1:
            count -= 1

        return count or 1

    def count_complex_words(self, words):
        return sum(1 for word in words if self.count_syllables(word) >= 3)

    def load_dale_chall_wordlist(self):
        with open("dale-chall-3000-words.txt", "r", encoding="utf-8") as f:
            words = set(line.strip().lower() for line in f if line.strip())
        return words
    
    def is_no_number(self, s):
        try:
            float(s)
            return False
        except ValueError:
            return True


    def count_difficult_words(self, words):
        return sum(1 for word in words if(word.lower() not in self.load_dale_chall_wordlist() and self.is_no_number(word)))

        
    def flesch(self, text): 

        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
        sentence_count = len(sentences)

        words = re.findall(r'\b(?:https?://\S+|[a-zA-Z0-9_/]+)\b', text) 
        word_count = len(words)


        syllable_count = sum(self.count_syllables(w) for w in words)

        if sentence_count == 0 or word_count == 0:
            return 0

        # Berechnung
        score = 0.39 * (word_count / sentence_count) + 11.8 * (syllable_count / word_count) - 15.59
        return score

        
    def gunning(self, text):
 
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
        sentence_count = len(sentences)

        words = re.findall(r'\b(?:https?://\S+|[a-zA-Z0-9_/]+)\b', text) 
        word_count = len(words)

        complex_word_count = self.count_complex_words(words)
       

        # Berechnung 

        score = 0.4 * ((word_count/sentence_count) + 100 * (complex_word_count/word_count))
        return score

        
    def dale_chall(self, text):

        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
        sentence_count = len(sentences)

        words = re.findall(r'\b(?:https?://\S+|[a-zA-Z0-9_/]+)\b', text) 
        word_count = len(words)

        complex_words_count = self.count_difficult_words(words)

        # Berechnung 
        score = 0.1579 * ((complex_words_count/word_count) * 100) + 0.0496 * (word_count/sentence_count)
        return score




