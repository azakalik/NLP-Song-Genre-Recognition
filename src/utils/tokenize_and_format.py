import nltk
import re
from typing import List
nltk.download('punkt')


def tokenize_and_format(text_to_tokenize: str, min_word_len: int | None) -> List[str]:
    # Tokenize the string into words
    tokens = nltk.word_tokenize(text_to_tokenize)

    # Remove non-alphabetic tokens, such as punctuation
    # We remove everything except letters and spaces
    if not min_word_len:
        words = [word.lower() for word in tokens if word.isalpha() or word.isspace()]
    else:
        words = [word.lower() for word in tokens if word.isalpha() and len(word) >= min_word_len]
    return words
