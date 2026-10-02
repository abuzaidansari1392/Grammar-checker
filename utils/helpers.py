"""
Utility helpers for the Grammar Checker application.
"""

import re
import io
from datetime import datetime
from data.grammar_dataset import ERROR_DESCRIPTIONS, ERROR_EXPLANATIONS


# ------------------------------------------------------------------ #
#  Text utilities
# ------------------------------------------------------------------ #

def count_words(text: str) -> int:
    return len(text.split())


def count_sentences(text: str) -> int:
    # Split on sentence-ending punctuation
    return len(re.findall(r'[.!?]+', text)) or 1


def compute_readability(text: str) -> dict:
    """
    Return basic readability stats (Flesch–Kincaid approximation).
    """
    words     = text.split()
    sentences = count_sentences(text)
    syllables = sum(_count_syllables(w) for w in words)

    n_words = max(len(words), 1)
    n_sents = max(sentences, 1)

    flesch = 206.835 - 1.015 * (n_words / n_sents) - 84.6 * (syllables / n_words)
    flesch = max(0, min(100, flesch))

    return {
        "words"     : n_words,
        "sentences" : n_sents,
        "syllables" : syllables,
        "avg_words_per_sentence": round(n_words / n_sents, 1),
        "flesch_score": round(flesch, 1),
        "readability_level": _flesch_level(flesch),
    }


def _count_syllables(word: str) -> int:
    word = word.lower().strip(".,!?;:'\"")
    if not word:
        return 0
    vowels  = "aeiou"
    count   = 0
    prev_v  = False
    for ch in word:
        is_v = ch in vowels
        if is_v and not prev_v:
            count += 1
        prev_v = is_v
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def _flesch_level(score: float) -> str:
    if score >= 90:
        return "Very Easy"
    if score >= 80:
        return "Easy"
    if score >= 70:
        return "Fairly Easy"
    if score >= 60:
        return "Standard"
    if score >= 50:
        return "Fairly Difficult"
    if score >= 30:
        return "Difficult"
    return "Very Difficult"


# ------------------------------------------------------------------ #
#  Error summary helpers
# ------------------------------------------------------------------ #

def summarize_errors(errors: list) -> dict:
    """
    Return a count of each error type and a list of unique types.
    """
    counts = {}
    for err in errors:
        counts[err.error_type] = counts.get(err.error_type, 0) + 1

    return {
        "total"      : len(errors),
        "by_type"    : counts,
        "types_found": list(counts.keys()),
        "descriptions": {
            k: ERROR_DESCRIPTIONS.get(k, k)
            for k in counts
        },
    }


def get_error_explanation(error_type: str) -> str:
    return ERROR_EXPLANATIONS.get(error_type, "No explanation available.")


def get_error_description(error_type: str) -> str:
    return ERROR_DESCRIPTIONS.get(error_type, error_type.replace("_", " ").title())


# ------------------------------------------------------------------ #
#  Download helpers
# ------------------------------------------------------------------ #

def generate_report(
    original: str,
    corrected: str,
    errors: list,
    grammar_score: float,
    stats: dict,
) -> str:
    """
    Generate a plain-text analysis report.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "=" * 60,
        "  GRAMMAR CHECK REPORT",
        f"  Generated: {now}",
        "=" * 60,
        "",
        "ORIGINAL TEXT:",
        "-" * 40,
        original,
        "",
        "CORRECTED TEXT:",
        "-" * 40,
        corrected,
        "",
        f"GRAMMAR SCORE: {grammar_score}/100",
        "",
        "STATISTICS:",
        "-" * 40,
        f"  Total Errors   : {stats.get('total', 0)}",
        f"  Word Count     : {stats.get('words', 0)}",
        f"  Sentence Count : {stats.get('sentences', 0)}",
        "",
    ]

    if errors:
        lines += ["ERRORS FOUND:", "-" * 40]
        for i, err in enumerate(errors, 1):
            lines += [
                f"\n[{i}] {get_error_description(err.error_type)}",
                f"    Text    : '{err.bad_text}'",
                f"    Message : {err.message}",
                f"    Fix     : {err.replacements[0] if err.replacements else 'N/A'}",
                f"    Explain : {get_error_explanation(err.error_type)}",
            ]
    else:
        lines += ["No errors found!"]

    lines += ["", "=" * 60]
    return "\n".join(lines)
