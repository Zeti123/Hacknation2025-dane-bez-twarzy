import spacy

from pipes.religion import add_religion_entity_ruler
from pipes.sex import add_sex_entity_ruler
from preprocessor import SpacyPreprocessor

LINES_PER_CHUNK = 100

if __name__ == '__main__':
    nlp = spacy.load("pl_core_news_md")

    nlp = add_sex_entity_ruler(nlp)
    nlp = add_religion_entity_ruler(nlp)

    with open("test_data.txt", "rt", encoding="utf-8") as file:
        text = file.read()
        preprocessor = SpacyPreprocessor(nlp)
        lines = text.splitlines()
        chunks = [lines[i:i + LINES_PER_CHUNK] for i in range(0, len(lines), LINES_PER_CHUNK)]
        for chunk in chunks:
            chunk_text = '\n'.join(chunk)
            result = preprocessor(chunk_text)
            print("--- ENTITIES ---")
            for e in result.entities:
                print(e.label, "->", e.text)
