"""
Grammar Detector Module
Uses LanguageTool + spaCy rule-based checks to detect grammar errors.
"""

import re
from dataclasses import dataclass, field


@dataclass
class GrammarError:
    """Represents a single grammar error."""
    error_type:  str
    message:     str
    offset:      int
    length:      int
    bad_text:    str
    replacements: list
    rule_id:     str = ""
    confidence:  float = 0.9

    @property
    def end_offset(self):
        return self.offset + self.length


# --------------------------------------------------------------------------- #
#  LanguageTool wrapper
# --------------------------------------------------------------------------- #

_lt_instance = None

def _get_lt():
    global _lt_instance
    if _lt_instance is None:
        try:
            import language_tool_python
            _lt_instance = language_tool_python.LanguageTool('en-US')
        except Exception as e:
            print(f"LanguageTool init failed: {e}")
            _lt_instance = None
    return _lt_instance


def detect_with_languagetool(text: str) -> list[GrammarError]:
    """Run LanguageTool and return a list of GrammarError objects."""
    lt = _get_lt()
    if lt is None:
        return []

    try:
        matches = lt.check(text)
    except Exception:
        return []

    errors = []
    for m in matches:
        # Support both old (ruleId) and new (rule_id) attribute names
        rule_id   = getattr(m, "rule_id",    getattr(m, "ruleId",    ""))
        message   = getattr(m, "message",    "")
        offset    = getattr(m, "offset",     0)
        err_len   = getattr(m, "errorLength", getattr(m, "error_length", 0))
        replacements = list(getattr(m, "replacements", [])[:5])

        error_type = _classify_lt_rule(rule_id, message)
        errors.append(GrammarError(
            error_type   = error_type,
            message      = message,
            offset       = offset,
            length       = err_len,
            bad_text     = text[offset: offset + err_len],
            replacements = replacements,
            rule_id      = rule_id,
            confidence   = 0.92,
        ))
    return errors


def _classify_lt_rule(rule_id: str, message: str) -> str:
    """Map a LanguageTool rule ID to one of our error categories."""
    rule_id = rule_id.upper()
    message = message.lower()

    if any(k in rule_id for k in ["AGREEMENT", "SVA", "SUBJECT_VERB"]):
        return "subject_verb"
    if any(k in rule_id for k in ["TENSE", "VERB_FORM"]):
        return "tense"
    if any(k in rule_id for k in ["ARTICLE", "DT_", "A_AN"]):
        return "article"
    if any(k in rule_id for k in ["PREP", "PREPOSITION"]):
        return "preposition"
    if any(k in rule_id for k in ["PRON", "PRONOUN"]):
        return "pronoun"
    if any(k in rule_id for k in ["SPELL", "MORFOLOGIK", "TYPO"]):
        return "spelling"
    if any(k in rule_id for k in ["PUNCT", "COMMA", "APOSTROPHE", "SENT_START"]):
        return "punctuation"
    if any(k in rule_id for k in ["WORD_ORDER", "ORDER"]):
        return "word_order"

    # Fallback to message keywords
    if "spell" in message or "misspell" in message:
        return "spelling"
    if "article" in message or '"a"' in message or '"an"' in message:
        return "article"
    if "verb" in message and "agree" in message:
        return "subject_verb"
    if "tense" in message:
        return "tense"
    if "preposition" in message:
        return "preposition"
    if "pronoun" in message:
        return "pronoun"
    if "comma" in message or "apostrophe" in message:
        return "punctuation"

    return "spelling"   # safe default


# --------------------------------------------------------------------------- #
#  spaCy rule-based checks
# --------------------------------------------------------------------------- #

def detect_with_spacy(text: str, doc=None) -> list[GrammarError]:
    """Apply rule-based grammar checks using spaCy."""
    if doc is None:
        from nlp.preprocessing import get_nlp
        doc = get_nlp()(text)

    errors: list[GrammarError] = []

    for sent in doc.sents:
        errors.extend(_check_subject_verb(sent, text))
        errors.extend(_check_article(sent, text))
        errors.extend(_check_double_negation(sent, text))

    return errors


def _check_subject_verb(sent, text: str) -> list[GrammarError]:
    """Detect obvious subject-verb disagreement."""
    errors = []
    tokens = list(sent)

    for i, token in enumerate(tokens):
        # "he/she/it" + base-form verb that should be 3rd-person singular
        if token.lower_ in ("he", "she", "it") and i + 1 < len(tokens):
            nxt = tokens[i + 1]
            # e.g. "he go" — verb is VBP (present non-3rd-person)
            if nxt.tag_ == "VBP" and nxt.lemma_ not in ("be", "have"):
                errors.append(GrammarError(
                    error_type   = "subject_verb",
                    message      = f"'{token.text} {nxt.text}' — verb should agree with singular subject.",
                    offset       = token.idx,
                    length       = nxt.idx + len(nxt.text) - token.idx,
                    bad_text     = token.text + " " + nxt.text,
                    replacements = [token.text + " " + nxt.lemma_ + "s"],
                    confidence   = 0.8,
                ))

        # plural noun + "was"
        if token.tag_ == "NNS" and i + 1 < len(tokens):
            nxt = tokens[i + 1]
            if nxt.lower_ == "was":
                errors.append(GrammarError(
                    error_type   = "subject_verb",
                    message      = f"Plural subject '{token.text}' should use 'were', not 'was'.",
                    offset       = nxt.idx,
                    length       = len(nxt.text),
                    bad_text     = nxt.text,
                    replacements = ["were"],
                    confidence   = 0.85,
                ))

    return errors


def _check_article(sent, text: str) -> list[GrammarError]:
    """Detect 'a' before vowel sounds and 'an' before consonant sounds."""
    errors = []
    tokens = list(sent)
    vowel_sounds = set("aeiouAEIOU")

    for i, token in enumerate(tokens):
        if token.lower_ not in ("a", "an") or i + 1 >= len(tokens):
            continue
        nxt = tokens[i + 1]
        nxt_word = nxt.text
        if not nxt_word:
            continue

        starts_with_vowel_sound = nxt_word[0] in vowel_sounds
        # Exceptions: "university", "unique", "European", "one", "hour"
        silent_h_words = {"hour", "honest", "honor", "heir"}
        consonant_u_words = {"university", "union", "unique", "unit",
                             "united", "uniform", "universal", "unity", "use"}

        if nxt_word.lower() in consonant_u_words:
            starts_with_vowel_sound = False
        if nxt_word.lower() in silent_h_words:
            starts_with_vowel_sound = True

        if token.lower_ == "a" and starts_with_vowel_sound:
            errors.append(GrammarError(
                error_type   = "article",
                message      = f"Use 'an' before words starting with a vowel sound ('{nxt_word}').",
                offset       = token.idx,
                length       = len(token.text),
                bad_text     = token.text,
                replacements = ["an"],
                confidence   = 0.9,
            ))
        elif token.lower_ == "an" and not starts_with_vowel_sound:
            errors.append(GrammarError(
                error_type   = "article",
                message      = f"Use 'a' before words starting with a consonant sound ('{nxt_word}').",
                offset       = token.idx,
                length       = len(token.text),
                bad_text     = token.text,
                replacements = ["a"],
                confidence   = 0.9,
            ))

    return errors


def _check_double_negation(sent, text: str) -> list[GrammarError]:
    """Flag obvious double negatives."""
    neg_words = {"no", "not", "never", "nobody", "nothing", "nowhere",
                 "neither", "nor", "hardly", "scarcely", "barely"}
    tokens = list(sent)
    neg_indices = [i for i, t in enumerate(tokens) if t.lower_ in neg_words]

    errors = []
    if len(neg_indices) >= 2:
        first = tokens[neg_indices[0]]
        errors.append(GrammarError(
            error_type   = "word_order",
            message      = "Double negative detected. In standard English, use only one negative per clause.",
            offset       = first.idx,
            length       = len(first.text),
            bad_text     = first.text,
            replacements = [],
            confidence   = 0.7,
        ))
    return errors


# --------------------------------------------------------------------------- #
#  Spelling checker
# --------------------------------------------------------------------------- #

def detect_spelling_errors(text: str) -> list[GrammarError]:
    """Use pyspellchecker to find misspelled words."""
    try:
        from spellchecker import SpellChecker
    except ImportError:
        return []

    spell = SpellChecker()
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    misspelled = spell.unknown(words)

    errors = []
    for word in misspelled:
        # find offset in original text (first occurrence)
        match = re.search(r'\b' + re.escape(word) + r'\b', text, re.IGNORECASE)
        if not match:
            continue
        candidates = list(spell.candidates(word) or [])
        errors.append(GrammarError(
            error_type   = "spelling",
            message      = f"'{word}' may be misspelled.",
            offset       = match.start(),
            length       = len(word),
            bad_text     = word,
            replacements = candidates[:5],
            rule_id      = "SPELL",
            confidence   = 0.85,
        ))
    return errors


# --------------------------------------------------------------------------- #
#  Combined detection
# --------------------------------------------------------------------------- #

def detect_all_errors(text: str, doc=None) -> list[GrammarError]:
    """
    Run all error detectors and return a de-duplicated, sorted list.
    """
    all_errors: list[GrammarError] = []

    # 1. LanguageTool
    all_errors.extend(detect_with_languagetool(text))

    # 2. spaCy rules
    all_errors.extend(detect_with_spacy(text, doc))

    # 3. Spelling (only add if not already flagged by LT)
    lt_offsets = {e.offset for e in all_errors if e.error_type == "spelling"}
    for err in detect_spelling_errors(text):
        if err.offset not in lt_offsets:
            all_errors.append(err)

    # De-duplicate by (offset, error_type)
    seen: set = set()
    unique: list[GrammarError] = []
    for e in all_errors:
        key = (e.offset, e.error_type)
        if key not in seen:
            seen.add(key)
            unique.append(e)

    return sorted(unique, key=lambda e: e.offset)


def compute_grammar_score(text: str, errors: list[GrammarError]) -> float:
    """
    Return a 0–100 grammar score based on error density.
    """
    words = len(text.split())
    if words == 0:
        return 100.0
    penalty = sum(1 for e in errors if e.error_type != "correct")
    score = max(0.0, 100.0 - (penalty / words) * 100.0)
    return round(score, 1)
