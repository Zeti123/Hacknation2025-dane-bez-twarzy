import time

import spacy

from labeling.pipes.age import add_age_entity_ruler
from labeling.pipes.keywords import add_keyword_entity_ruler
from labeling.pipes.relative import add_relative_entity_ruler
from labeling.pipes.religion import add_religion_entity_ruler
from labeling.pipes.rule_entities import add_rule_entity_ruler
from labeling.pipes.sex import add_sex_entity_ruler
from labeling.preprocessor import SpacyPreprocessor


def anonymize(text: str) -> str:
    nlp = spacy.load("pl_core_news_md")
    nlp.max_length = max(nlp.max_length, 2_000_000)

    # Rule-based patterns first to capture regex-like matches via EntityRuler
    nlp = add_rule_entity_ruler(nlp)
    nlp = add_keyword_entity_ruler(nlp)
    nlp = add_sex_entity_ruler(nlp)
    nlp = add_religion_entity_ruler(nlp)
    nlp = add_relative_entity_ruler(nlp)
    nlp = add_age_entity_ruler(nlp)

    preprocessor = SpacyPreprocessor(nlp)

    start_time = time.time()
    result = preprocessor(text)
    print(f"--- Anonymization took {time.time() - start_time} seconds ---")

    return result.redacted_text