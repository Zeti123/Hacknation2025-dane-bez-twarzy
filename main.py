import spacy
from spacy_processing import SpacyPreprocessor
from labeling.regex_labeling import *

try:
    nlp = spacy.load("pl_core_news_md")
except OSError as e:
    raise RuntimeError(
        "Polish model 'pl_core_news_md' is not installed. "
        "Run: python -m spacy download pl_core_news_md"
    ) from e

nlp

preprocessor = SpacyPreprocessor()

example_text = """
Nazywam się Jan Kowalski, moim ulubionym aktorem jest Maciej Musiał.
Mieszkam w Warszawie przy ulicy Długiej 5.
ostatnio brałem ślub 15 listopada 2025.
Nazywam się Jan Kowalski, mój niepoprawny PESEL to 90010112345 a poprawny to 02070803628.
Mieszkam w Warszawie przy ulicy Długiej 5. Numer karty to np. 4111-1111-1111-1111 a dowód ABC123456 bartolomeo123@gmail.com
665-333-785
"""

result = preprocessor(example_text)

result.meta

for t in result.tokens[:20]:
    print(
        f"{t.idx:>2}: {t.text:<15} POS={t.pos:<5} TAG={t.tag:<8} MORPH={t.morph}"
    )

result.entities += regex_labeling(example_text)

for e in result.entities:
    print(f"[{e.label}] {e.text} ({e.start_char}-{e.end_char})")
