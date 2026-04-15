"""
preprocess.py — Text Preprocessing Module for Email Spam Detection
==================================================================
Handles all NLP preprocessing: cleaning, tokenization, stopword removal,
stemming, and feature extraction via TF-IDF.
"""

import re
import string
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Download required NLTK data (silent) ────────────────────────────────────
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

# ── Globals ──────────────────────────────────────────────────────────────────
_stemmer = PorterStemmer()
_stop_words = set(stopwords.words("english"))


# ── Core text-cleaning function ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Apply a pipeline of cleaning steps to a single text string:
      1. Lowercase
      2. Remove URLs
      3. Remove HTML tags
      4. Remove digits
      5. Remove punctuation
      6. Tokenize
      7. Remove stopwords
      8. Apply Porter stemming
      9. Rejoin tokens

    Parameters
    ----------
    text : str
        Raw email / message text.

    Returns
    -------
    str
        Cleaned, stemmed text ready for vectorization.
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove digits
    text = re.sub(r"\d+", "", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords & stem
    cleaned_tokens = [
        _stemmer.stem(word)
        for word in tokens
        if word not in _stop_words and len(word) > 1
    ]

    return " ".join(cleaned_tokens)


# ── DataFrame-level preprocessing ───────────────────────────────────────────
def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accept a raw DataFrame, detect the text & label columns, clean
    text, and encode labels (spam → 1, ham → 0).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least a text column and a label column.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with columns ['text', 'label', 'cleaned_text'].

    Raises
    ------
    ValueError
        If the expected columns cannot be identified.
    """
    df = df.copy()

    # ── Auto-detect columns ──────────────────────────────────────────────
    # Common column name patterns for popular spam datasets
    col_map = {}
    cols_lower = {c.lower().strip(): c for c in df.columns}

    # Label column detection
    for candidate in ["v1", "label", "class", "category", "target", "spam"]:
        if candidate in cols_lower:
            col_map["label"] = cols_lower[candidate]
            break

    # Text column detection
    for candidate in ["v2", "text", "message", "email", "sms", "content", "body"]:
        if candidate in cols_lower:
            col_map["text"] = cols_lower[candidate]
            break

    if "label" not in col_map or "text" not in col_map:
        raise ValueError(
            "Could not auto-detect text and label columns. "
            "Please ensure your CSV has columns named like 'text'/'message' "
            "and 'label'/'class'/'category'."
        )

    # Rename to standard names
    df = df.rename(columns={col_map["label"]: "label", col_map["text"]: "text"})
    df = df[["text", "label"]].dropna()

    # ── Encode labels ────────────────────────────────────────────────────
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["label"] = df["label"].map({"spam": 1, "ham": 0})
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # ── Clean text ───────────────────────────────────────────────────────
    df["cleaned_text"] = df["text"].apply(clean_text)

    return df.reset_index(drop=True)


# ── TF-IDF Vectorizer builder ───────────────────────────────────────────────
def build_tfidf_vectorizer(corpus, max_features: int = 5000):
    """
    Fit a TF-IDF vectorizer on the training corpus and return both the
    vectorizer and the transformed matrix.

    Parameters
    ----------
    corpus : iterable of str
        Cleaned text documents.
    max_features : int
        Maximum vocabulary size.

    Returns
    -------
    vectorizer : TfidfVectorizer
        Fitted vectorizer (needed at prediction time).
    X : sparse matrix
        TF-IDF feature matrix.
    """
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X = vectorizer.fit_transform(corpus)
    return vectorizer, X


# ── Sample / fallback dataset ───────────────────────────────────────────────
def get_sample_dataset() -> pd.DataFrame:
    """
    Return a small built-in sample dataset so the app can demonstrate
    functionality even without an uploaded CSV.
    """
    data = {
        "text": [
            "Congratulations! You've won a $1000 gift card. Call now!",
            "Hey, are we still meeting for lunch tomorrow?",
            "URGENT: Your account has been compromised. Click here immediately.",
            "Can you pick up some milk on your way home?",
            "FREE entry in a weekly competition to win an iPad!",
            "I'll be there in 10 minutes, just stuck in traffic.",
            "You have been selected for a cash prize! Claim now.",
            "Don't forget mom's birthday next week.",
            "Act now! Limited time offer on weight loss pills.",
            "Meeting rescheduled to 3pm. See you there.",
            "Win a brand new car! Text WIN to 12345 now!",
            "Thanks for dinner last night, it was great!",
            "WINNER!! You've been selected for exclusive rewards!",
            "Can you send me the report by end of day?",
            "Claim your free holiday now! Call 0800-123-456.",
            "Let's grab coffee this afternoon if you're free.",
            "Your loan has been approved! No credit check needed.",
            "The kids loved the theme park yesterday.",
            "Get rich quick! Invest now with guaranteed returns!",
            "I'll call you back after the meeting ends.",
            "FLASH SALE: 90% OFF designer watches! Buy now!",
            "See you at the gym tomorrow morning.",
            "You are a winner! Collect your prize money today!",
            "Happy birthday! Hope you have a wonderful day.",
            "Make money from home! No experience required!",
            "Dinner at 7? I'll book the restaurant.",
            "FREE ringtones and wallpapers! Text GO to 87121.",
            "Good morning! Don't forget the team standup at 9.",
            "Exclusive deal: Credit card with 0% APR. Apply now!",
            "The presentation went really well today.",
        ],
        "label": [
            "spam", "ham", "spam", "ham", "spam",
            "ham", "spam", "ham", "spam", "ham",
            "spam", "ham", "spam", "ham", "spam",
            "ham", "spam", "ham", "spam", "ham",
            "spam", "ham", "spam", "ham", "spam",
            "ham", "spam", "ham", "spam", "ham",
        ],
    }
    return pd.DataFrame(data)
