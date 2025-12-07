import spacy
from time import time
from pipes.age import add_age_entity_ruler
from pipes.keywords import add_keyword_entity_ruler
from pipes.relative import add_relative_entity_ruler
from pipes.religion import add_religion_entity_ruler
from pipes.rule_entities import add_rule_entity_ruler
from pipes.sex import add_sex_entity_ruler
from preprocessor import SpacyPreprocessor


def main():
    """Run labeling over full input text and emit entities + redacted text."""
    nlp = spacy.load("pl_core_news_md")
    nlp.max_length = max(nlp.max_length, 2_000_000)

    # Rule-based patterns first to capture regex-like matches via EntityRuler
    nlp = add_rule_entity_ruler(nlp)
    nlp = add_keyword_entity_ruler(nlp)
    nlp = add_sex_entity_ruler(nlp)
    nlp = add_religion_entity_ruler(nlp)
    nlp = add_relative_entity_ruler(nlp)
    nlp = add_age_entity_ruler(nlp)

    with open("./labeling/test_data.txt", "rt", encoding="utf-8") as file:
        text = file.read()
    start = time.now()
    preprocessor = SpacyPreprocessor(nlp)
    result = preprocessor(text)

    print("--- ENTITIES ---")
    for e in result.entities:
        print(e.label, "->", e.text)

    print("--- REDACTED ---")
    print(result.redacted_text)

    # Persist unique label-text pairs for quick inspection.
    seen = set()
    with open("./labeling/labeled_entities.txt", "wt", encoding="utf-8") as out_file:
        for entity in result.entities:
            key = (entity.label, entity.text)
            if key in seen:
                continue
            seen.add(key)
            out_file.write(f"{entity.label}\t{entity.text}\n")


if __name__ == '__main__':
    main()
