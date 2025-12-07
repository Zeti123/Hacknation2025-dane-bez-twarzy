from spacy_processing import SpacyPreprocessor
from context_manager import ContextManager
from ollama_classifier import OllamaEntityClassifier
from classes import EntityHint
from format import apply_placeholders


with open('text.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()

pre = SpacyPreprocessor()
pre_result = pre(raw_text)
print(pre_result.meta)


cm = ContextManager()
chunks = cm.chunk_by_hints(pre_result, hints=pre_result.entities, sentence_radius=0)

for i, ch in enumerate(chunks):
    print(f"Chunk {i}: chars {ch.start_char}-{ch.end_char}, sentences={len(ch.sentences)}, entities={len(ch.entities)}")
    print(ch.text)
    print('---')


classifier = OllamaEntityClassifier()

try:
    classified = classifier.classify_document(chunks,verbose=True)
except Exception as exc:
    print("Nie udało się wywołać Ollama (uruchom serwer na localhost:11434):", exc)
else:
    for (start, end), label in sorted(classified.items()):
        span = raw_text[start:end]
        status = 'OK' if label != 'none' else 'ODRZUCONE'
        print(f"{status:10} {label:18} -> {span!r}")

if 'classified' in locals():
    redacted = apply_placeholders(raw_text, classified)
    print(redacted)
