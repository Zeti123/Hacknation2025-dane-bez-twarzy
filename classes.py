from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class TokenInfo:
    idx: int                 # token index in doc
    text: str
    lemma: str
    pos: str                 # coarse POS tag
    tag: str                 # detailed tag
    morph: str               # raw morph string
    dep: str                 # dependency relation
    head: int                # index of head token
    is_stop: bool
    is_punct: bool
    whitespace: str          # trailing whitespace


@dataclass
class SentenceInfo:
    sent_id: int
    text: str
    start_char: int
    end_char: int
    token_indices: List[int]  # indices of tokens belonging to this sentence


@dataclass
class SentenceSpan:
    text: str
    start_char: int
    end_char: int
    est_tokens: int


@dataclass
class EntityHint:
    text: str
    label: str
    start_char: int
    end_char: int


@dataclass
class Chunk:
    text: str
    start_char: int
    end_char: int
    sentences: List[SentenceSpan]
    entities: List[EntityHint]


@dataclass
class PreprocessResult:
    raw_text: str
    tokens: List[TokenInfo]
    sentences: List[SentenceInfo]
    entities: List[EntityHint]
    meta: Dict[str, Any]


def estimate_tokens_from_chars(text: str, chars_per_token: float = 4.0) -> int:
    return max(1, int(len(text) / chars_per_token))


ALLOWED_LABELS = [
    "name",
    "surname",
    "age",
    "date-of-birth",
    "date",
    "sex",
    "religion",
    "political-view",
    "ethnicity",
    "sexual-orientation",
    "health",
    "relative",
    "city",
    "address",
    "email",
    "phone",
    "pesel",
    "document-number",
    "company",
    "school-name",
    "job-title",
    "bank-account",
    "credit-card-number",
    "username",
    "secret",
    "none",  # for entities that should be removed but don't fit other categories
]
