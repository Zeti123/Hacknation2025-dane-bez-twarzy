import spacy
from spacy_processing import SpacyPreprocessor
try:
    nlp = spacy.load("pl_core_news_md")
except OSError as e:
    raise RuntimeError(
        "Polish model 'pl_core_news_md' is not installed. "
        "Run: python -m spacy download pl_core_news_md"
    ) from e

nlp

example_text = """
Nazywam się Jan Kowalski, mój PESEL to 90010112345.
Mieszkam w Warszawie przy ulicy Długiej 5.
"""

preprocessor = SpacyPreprocessor()
result = preprocessor(example_text)

result.meta
