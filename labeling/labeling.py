import spacy

from pipes.relative import add_relative_entity_ruler
from pipes.religion import add_religion_entity_ruler
from pipes.sex import add_sex_entity_ruler
from pipes.age import add_age_entity_ruler
from pipes.date import add_date_entity_ruler
from pipes.keywords import add_keyword_entity_ruler
from preprocessor import SpacyPreprocessor
from regex_labeling import regex_labeling

LINES_PER_CHUNK = 100

if __name__ == '__main__':
    nlp = spacy.load("pl_core_news_md")

    nlp = add_keyword_entity_ruler(nlp)
    nlp = add_sex_entity_ruler(nlp)
    nlp = add_religion_entity_ruler(nlp)
    nlp = add_relative_entity_ruler(nlp)
    nlp = add_age_entity_ruler(nlp)
    nlp = add_date_entity_ruler(nlp)


    with open("./labeling/test_data.txt", "rt", encoding="utf-8") as file:
        text = file.read()
        preprocessor = SpacyPreprocessor(nlp)
        lines = text.splitlines()
        rgx = regex_labeling(text)
        results =  set(rgx)
        chunks = [lines[i:i + LINES_PER_CHUNK] for i in range(0, len(lines), LINES_PER_CHUNK)]
        for chunk in chunks:
            chunk_text = '\n'.join(chunk)
            result = preprocessor(chunk_text)
            results.update(result.entities)
            # print("--- ENTITIES ---")
            # for e in result.entities:
            #     print(e.label, "->", e.text)
            # print("--- REDACTED ---")
            # print(result.redacted_text)
    
    with open("./labeling/labeled_entities.txt", "wt", encoding="utf-8") as out_file:
        for entity in results:
            out_file.write(f"{entity.label}\t{entity.text}\n ")


