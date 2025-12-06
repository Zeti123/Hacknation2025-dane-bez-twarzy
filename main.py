from parser import *

preprocessor = SpacyPreprocessor()

example_text = """
Nazywam się Jan Kowalski, moim ulubionym aktorem jest Maciej Musiał.
Mieszkam w Warszawie przy ulicy Długiej 5.
"""

result = preprocessor(example_text)

result.meta

for t in result.tokens[:20]:
    print(
        f"{t.idx:>2}: {t.text:<15} POS={t.pos:<5} TAG={t.tag:<8} MORPH={t.morph}"
    )

for e in result.entities:
    print(f"[{e.label}] {e.text} ({e.start_char}-{e.end_char})")