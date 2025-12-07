import time
from typing import Optional

import spacy

from labeling.pipes.age import add_age_entity_ruler
from labeling.pipes.keywords import add_keyword_entity_ruler
from labeling.pipes.relative import add_relative_entity_ruler
from labeling.pipes.religion import add_religion_entity_ruler
from labeling.pipes.rule_entities import add_rule_entity_ruler
from labeling.pipes.sex import add_sex_entity_ruler
from labeling.preprocessor import PreprocessResult, SpacyPreprocessor

DEFAULT_MODEL = "pl_core_news_md"
DEFAULT_MAX_LEN = 2_000_000


def build_pipeline(model: str = DEFAULT_MODEL, max_length: int = DEFAULT_MAX_LEN) -> spacy.language.Language:
    """
    Build and configure the spaCy pipeline with rule-based entity rulers.
    """
    nlp = spacy.load(model)
    nlp.max_length = max(nlp.max_length, max_length)

    # Rule-based patterns first to capture regex-like matches via EntityRuler
    nlp = add_rule_entity_ruler(nlp)
    nlp = add_keyword_entity_ruler(nlp)
    nlp = add_sex_entity_ruler(nlp)
    nlp = add_religion_entity_ruler(nlp)
    nlp = add_relative_entity_ruler(nlp)
    nlp = add_age_entity_ruler(nlp)

    return nlp


def anonymize(
    text: str,
    *,
    model: str = DEFAULT_MODEL,
    max_length: int = DEFAULT_MAX_LEN,
    use_ner_hints: bool = True,
    verbose: bool = True,
    return_full: bool = False,
    nlp: Optional[spacy.language.Language] = None,
) -> str | PreprocessResult:
    """
    Run the anonymization pipeline on a raw text string.

    Args:
        text: Raw text to anonymize.
        model: spaCy model name to load (ignored if `nlp` is provided).
        max_length: Max document length override for spaCy.
        use_ner_hints: Whether to use spaCy NER hints in preprocessing.
        verbose: Print timing information when True.
        return_full: When True, return the full PreprocessResult; otherwise return the redacted text.
        nlp: Optional preloaded spaCy pipeline to reuse.
    """
    pipeline = nlp or build_pipeline(model=model, max_length=max_length)
    preprocessor = SpacyPreprocessor(pipeline, use_ner_hints=use_ner_hints)

    start_time = time.time()
    result = preprocessor(text)
    if verbose:
        print(f"--- Anonymization took {time.time() - start_time:.2f} seconds ---")

    return result if return_full else result.redacted_text
