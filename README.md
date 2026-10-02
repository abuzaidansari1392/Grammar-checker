# ✍️ NLP-Based English Grammar Checker

A complete, modular grammar checking and correction system built with Python, spaCy, LanguageTool, scikit-learn, and Streamlit.

---

## Features

| Feature | Tool |
|---|---|
| Tokenization, POS tagging, Lemmatization, Dependency parsing | spaCy |
| Grammar rule detection & correction | LanguageTool |
| ML-based error classification | TF-IDF + Logistic Regression |
| Spelling correction | pyspellchecker |
| Interactive UI | Streamlit |

**Detected Error Types**
- Subject-Verb Agreement
- Tense Errors
- Article Errors (a/an/the)
- Preposition Errors
- Pronoun Errors
- Spelling Errors
- Punctuation Errors
- Word Order Errors

---

## Project Structure

```
grammar-checker/
├── app.py                    # Streamlit UI (entry point)
├── train.py                  # Standalone model training script
├── requirements.txt
├── README.md
├── data/
│   ├── __init__.py
│   └── grammar_dataset.py    # Labelled training data & descriptions
├── models/                   # Saved ML models (auto-created)
├── nlp/
│   ├── __init__.py
│   ├── preprocessing.py      # spaCy NLP pipeline
│   ├── grammar_detector.py   # LanguageTool + rule-based detection
│   ├── error_classifier.py   # TF-IDF + LR classifier
│   └── correction.py         # Correction engine + HTML highlighting
└── utils/
    ├── __init__.py
    └── helpers.py            # Stats, readability, download utils
```

---

## Installation

### 1. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the spaCy English model

```bash
python -m spacy download en_core_web_sm
```

### 4. (Optional) Train the ML classifier explicitly

The app auto-trains on first launch, but you can pre-train:

```bash
python train.py
```

---

## Running the Application

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

---

## Dataset Format

Training data lives in `data/grammar_dataset.py` as a list of `(sentence, label)` tuples:

```python
TRAINING_DATA = [
    ("She don't like coffee.", "subject_verb"),
    ("Yesterday, I go to the market.", "tense"),
    ("She is a honest woman.", "article"),
    ("She is interested on music.", "preposition"),
    ("Him went to the store.", "pronoun"),
    ("She recieved a letter today.", "spelling"),
    ("Lets go to the park today", "punctuation"),
    ("Always she comes late.", "word_order"),
    ("She goes to school every day.", "correct"),
    # … add more rows to improve accuracy
]
```

**Labels:** `subject_verb | tense | article | preposition | pronoun | spelling | punctuation | word_order | correct`

---

## ML Model Details

| Component | Value |
|---|---|
| Vectoriser | TF-IDF (1–3 grams, max 8 000 features) |
| Classifier | Logistic Regression (C=5, balanced weights) |
| Train / Test split | 80 / 20, stratified |
| Metrics | Accuracy, Precision, Recall, F1 (weighted), Confusion Matrix |

The model is saved to `models/error_classifier.joblib` and loaded automatically by the app.

---

## Extending the System

- **Add training data** — append rows to `TRAINING_DATA` in `data/grammar_dataset.py`.
- **Add rule checks** — add functions in `nlp/grammar_detector.py`.
- **Swap the classifier** — replace `LogisticRegression` in `nlp/error_classifier.py` with any sklearn estimator.
- **Add a new error category** — add entries in `ERROR_DESCRIPTIONS` / `ERROR_EXPLANATIONS` (grammar_dataset.py) and the colour map in `app.py`.

---

## Requirements

- Python 3.9+
- Java 8+ (required by LanguageTool — auto-downloaded on first run)
- Internet connection on first run (for spaCy model + LanguageTool download)
