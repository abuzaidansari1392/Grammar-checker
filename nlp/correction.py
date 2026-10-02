"""
Grammar Correction Module
Applies corrections suggested by LanguageTool and rule-based fixes.
"""

import re
from nlp.grammar_detector import GrammarError


# ------------------------------------------------------------------ #
#  LanguageTool-based correction
# ------------------------------------------------------------------ #

def correct_with_languagetool(text: str) -> str:
    """
    Apply LanguageTool's top suggestion for every detected error.
    Falls back to the original text if LanguageTool is unavailable.
    """
    try:
        import language_tool_python
        tool = language_tool_python.LanguageTool('en-US')
        return language_tool_python.utils.correct(text, tool.check(text))
    except Exception:
        return text


# ------------------------------------------------------------------ #
#  Rule-based corrections
# ------------------------------------------------------------------ #

# Common spelling corrections (extend as needed)
_SPELLING_FIXES = {
    "recieve":      "receive",
    "recieved":     "received",
    "beleive":      "believe",
    "beleived":     "believed",
    "occured":      "occurred",
    "occurance":    "occurrence",
    "buisness":     "business",
    "beautifull":   "beautiful",
    "achived":      "achieved",
    "definately":   "definitely",
    "responsable":  "responsible",
    "seperated":    "separated",
    "greatful":     "grateful",
    "goverment":    "government",
    "embarassed":   "embarrassed",
    "embarass":     "embarrass",
    "recomended":   "recommended",
    "neccessarily": "necessarily",
    "wierd":        "weird",
    "occassion":    "occasion",
    "accomodate":   "accommodate",
    "noticable":    "noticeable",
    "untill":       "until",
    "reccommend":   "recommend",
    "tommorow":     "tomorrow",
    "tommorrow":    "tomorrow",
    "enviroment":   "environment",
    "existance":    "existence",
    "independance": "independence",
    "persistance":  "persistence",
    "prevelant":    "prevalent",
}

# a / an article fixes (word → correct article)
_ARTICLE_FIXES_AN = {
    "hour", "honest", "honor", "honours", "honorable", "heir",
}
_ARTICLE_FIXES_A = {
    "university", "union", "unique", "unit", "united", "uniform",
    "universal", "unity", "use", "user", "usual", "European", "euro",
    "one", "once",
}


def apply_rule_based_corrections(text: str) -> str:
    """
    Apply simple regex-based rule corrections:
    - Spelling fixes
    - a/an article fixes
    - Basic punctuation normalization
    """
    corrected = text

    # 1. Spelling
    for wrong, right in _SPELLING_FIXES.items():
        corrected = re.sub(r'\b' + wrong + r'\b', right, corrected, flags=re.IGNORECASE)

    # 2. a → an before vowel-sound words
    def fix_article(match):
        article = match.group(1)
        space   = match.group(2)
        word    = match.group(3)
        first   = word[0].lower()
        word_lc = word.lower()

        if word_lc in _ARTICLE_FIXES_A:
            correct = "a"
        elif word_lc in _ARTICLE_FIXES_AN:
            correct = "an"
        elif first in "aeiou":
            correct = "an"
        else:
            correct = "a"

        # Preserve capitalisation of original article
        if article[0].isupper():
            correct = correct.capitalize()
        return correct + space + word

    corrected = re.sub(
        r'\b(a|an)([\s]+)(\w+)',
        fix_article,
        corrected,
        flags=re.IGNORECASE,
    )

    # 3. Ensure sentence starts with capital
    if corrected and corrected[0].islower():
        corrected = corrected[0].upper() + corrected[1:]

    # 4. Ensure sentence ends with punctuation
    if corrected and corrected[-1] not in ".!?":
        corrected += "."

    return corrected


# ------------------------------------------------------------------ #
#  Apply corrections from detected GrammarError objects
# ------------------------------------------------------------------ #

def apply_corrections(text: str, errors: list[GrammarError]) -> str:
    """
    Apply the first suggested replacement for each error in reverse
    offset order (so earlier offsets aren't shifted).
    """
    # Sort errors in reverse offset order
    sorted_errors = sorted(errors, key=lambda e: e.offset, reverse=True)

    corrected = text
    for error in sorted_errors:
        if not error.replacements:
            continue
        replacement = error.replacements[0]
        start = error.offset
        end   = error.end_offset
        corrected = corrected[:start] + replacement + corrected[end:]

    return corrected


# ------------------------------------------------------------------ #
#  Highlight errors in HTML
# ------------------------------------------------------------------ #

_ERROR_COLORS = {
    "subject_verb": "#FF6B6B",
    "tense":        "#FFA726",
    "article":      "#AB47BC",
    "preposition":  "#42A5F5",
    "pronoun":      "#26A69A",
    "spelling":     "#EF5350",
    "punctuation":  "#78909C",
    "word_order":   "#8D6E63",
    "correct":      "#66BB6A",
}


def highlight_errors_html(text: str, errors: list[GrammarError]) -> str:
    """
    Return the text with error spans wrapped in colored <mark> tags.
    """
    if not errors:
        return text

    result = []
    prev = 0
    for error in sorted(errors, key=lambda e: e.offset):
        start = error.offset
        end   = error.end_offset
        color = _ERROR_COLORS.get(error.error_type, "#FF6B6B")

        result.append(text[prev:start])
        result.append(
            f'<mark style="background-color:{color}33; border-bottom: 2px solid {color}; '
            f'border-radius:3px; padding:1px 2px;" '
            f'title="{error.message}">{text[start:end]}</mark>'
        )
        prev = end

    result.append(text[prev:])
    return "".join(result)


# ------------------------------------------------------------------ #
#  Master correction pipeline
# ------------------------------------------------------------------ #

def full_correction_pipeline(text: str, errors: list[GrammarError]) -> dict:
    """
    Run all correction steps and return a summary dict.
    """
    # Step 1: LanguageTool automated correction
    lt_corrected   = correct_with_languagetool(text)

    # Step 2: Rule-based spelling / article / punctuation fixes
    rule_corrected = apply_rule_based_corrections(lt_corrected)

    # Step 3: HTML highlighting of original errors
    highlighted    = highlight_errors_html(text, errors)

    return {
        "original":    text,
        "corrected":   rule_corrected,
        "highlighted": highlighted,
        "lt_corrected": lt_corrected,
    }
