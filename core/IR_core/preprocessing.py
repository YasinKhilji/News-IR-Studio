from typing import List
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Ensure resources exist (callers should also download on setup)
try:
    _ = stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

try:
    nltk.data.find("corpora/omw-1.4")
except LookupError:
    nltk.download("omw-1.4")

EN_STOP = set(stopwords.words("english"))
STEMMER = PorterStemmer()
LEMMATIZER = WordNetLemmatizer()

_punct_re = re.compile(r"[^a-zA-Z0-9\s]")
_digits_re = re.compile(r"\d+")

def preprocess_text(text: str,
                    lowercase: bool = True,
                    remove_punct: bool = True,
                    remove_digits: bool = False,
                    remove_stopwords: bool = True,
                    use_stemming: bool = True,
                    use_lemmatize: bool = False,
                    min_token_len: int = 2,
                    max_token_len: int = 40) -> List[str]:
    if not isinstance(text, str):
        text = str(text)

    if lowercase:
        text = text.lower()

    if remove_punct:
        text = _punct_re.sub(" ", text)

    if remove_digits:
        text = _digits_re.sub(" ", text)

    tokens = [t for t in word_tokenize(text) if t.strip()]

    # length filter
    tokens = [t for t in tokens if min_token_len <= len(t) <= max_token_len]

    if remove_stopwords:
        tokens = [t for t in tokens if t not in EN_STOP]

    if use_lemmatize:
        tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    elif use_stemming:
        tokens = [STEMMER.stem(t) for t in tokens]

    return tokens
