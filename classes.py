from dataclasses import dataclass
from typing import List, Any, Dict

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
class EntityHint:
    text: str
    label: str
    start_char: int
    end_char: int


@dataclass
class PreprocessResult:
    raw_text: str
    tokens: List[TokenInfo]
    sentences: List[SentenceInfo]
    entities: List[EntityHint]
    redacted_text: str
    meta: Dict[str, Any]
