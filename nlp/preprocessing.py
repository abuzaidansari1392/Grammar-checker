"""
Text Preprocessing Module
Handles tokenization, POS tagging, lemmatization, and dependency parsing.
"""

import re
import string


def load_spacy_model():
    """Lazy-load spaCy model to avoid repeated loading."""
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except OSError:
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                       capture_output=True)
        import spacy
        return spacy.load("en_core_web_sm")


# Module-level cache
_nlp = None

def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = load_spacy_model()
    return _nlp


def preprocess_text(text: str) -> dict:
    """
    Full NLP pipeline: tokenize, POS-tag, lemmatize, dependency parse.

    Returns a dict with keys:
        original, cleaned, sentences, tokens, pos_tags,
        lemmas, dependencies, doc
    """
    # --- basic cleaning ---
    cleaned = text.strip()
    cleaned = re.sub(r'\s+', ' ', cleaned)   # collapse whitespace

    nlp = get_nlp()
    doc = nlp(cleaned)

    # sentence tokenization
    sentences = [sent.text.strip() for sent in doc.sents]

    # word tokens (exclude pure whitespace)
    tokens = [token.text for token in doc if not token.is_space]

    # POS tags
    pos_tags = [(token.text, token.pos_, token.tag_)
                for token in doc if not token.is_space]

    # lemmas
    lemmas = [(token.text, token.lemma_)
              for token in doc if not token.is_space]

    # dependency parse
    dependencies = [
        {
            "token": token.text,
            "dep":   token.dep_,
            "head":  token.head.text,
            "pos":   token.pos_,
        }
        for token in doc if not token.is_space
    ]

    return {
        "original":     text,
        "cleaned":      cleaned,
        "sentences":    sentences,
        "tokens":       tokens,
        "pos_tags":     pos_tags,
        "lemmas":       lemmas,
        "dependencies": dependencies,
        "doc":          doc,
    }


def tokenize_sentences(text: str) -> list[str]:
    """Return a list of sentences from the input text."""
    nlp = get_nlp()
    doc = nlp(text.strip())
    return [sent.text.strip() for sent in doc.sents]


def tokenize_words(text: str) -> list[str]:
    """Return a list of word tokens, excluding punctuation and spaces."""
    nlp = get_nlp()
    doc = nlp(text)
    return [t.text for t in doc if not t.is_punct and not t.is_space]


def get_pos_tags(text: str) -> list[tuple]:
    """Return (word, coarse_pos, fine_pos) tuples."""
    nlp = get_nlp()
    doc = nlp(text)
    return [(t.text, t.pos_, t.tag_) for t in doc if not t.is_space]


def lemmatize(text: str) -> list[tuple]:
    """Return (word, lemma) tuples."""
    nlp = get_nlp()
    doc = nlp(text)
    return [(t.text, t.lemma_) for t in doc if not t.is_space]


def get_dependency_parse(text: str) -> list[dict]:
    """Return dependency parse as a list of dicts."""
    nlp = get_nlp()
    doc = nlp(text)
    return [
        {"token": t.text, "dep": t.dep_, "head": t.head.text, "pos": t.pos_}
        for t in doc if not t.is_space
    ]


def extract_noun_phrases(text: str) -> list[str]:
    """Return noun chunks from text."""
    nlp = get_nlp()
    doc = nlp(text)
    return [chunk.text for chunk in doc.noun_chunks]


def extract_entities(text: str) -> list[tuple]:
    """Return (entity_text, entity_label) tuples."""
    nlp = get_nlp()
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]


def clean_text_for_ml(text: str) -> str:
    """
    Lightweight cleaning for ML feature extraction.
    Lowercases, removes extra whitespace.
    Keeps punctuation so the classifier can use it as a signal.
    """
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text
