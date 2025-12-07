from dataclasses import dataclass
from typing import Any, Dict, List

import spacy

from regex_labeling import regex_labeling


@dataclass
class TokenInfo:
    idx: int  # token index in doc
    text: str
    lemma: str
    pos: str  # coarse POS tag
    tag: str  # detailed tag
    dep: str  # dependency relation
    head: int  # index of head token
    is_stop: bool
    is_punct: bool
    whitespace: str  # trailing whitespace


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
    def __hash__(self):
        return hash((self.text, self.label, self.start_char, self.end_char))


@dataclass
class PreprocessResult:
    raw_text: str
    tokens: List[TokenInfo]
    sentences: List[SentenceInfo]
    entities: List[EntityHint]
    redacted_text: str
    meta: Dict[str, Any]


ALLOWED_LABELS = {
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
}

ORG_SCHOOL_KEYWORDS = (
    "szkoła",
    "liceum",
    "technikum",
    "uniwersytet",
    "akademia",
    "politechnika",
)


class SpacyPreprocessor:
    def __init__(
            self,
            nlp: spacy.Language,
            use_ner_hints: bool = True,
            use_regex_hints: bool = True,
    ) -> None:
        self.nlp = nlp
        self.use_ner_hints = use_ner_hints
        self.use_regex_hints = use_regex_hints

    def _tokens_to_info(self, doc: spacy.language.Doc) -> List[TokenInfo]:
        tokens_info: List[TokenInfo] = []
        for i, token in enumerate(doc):
            tokens_info.append(
                TokenInfo(
                    idx=i,
                    text=token.text,
                    lemma=token.lemma_,
                    pos=token.pos_,
                    tag=token.tag_,
                    dep=token.dep_,
                    head=token.head.i,
                    is_stop=token.is_stop,
                    is_punct=token.is_punct,
                    whitespace=token.whitespace_,
                )
            )
        return tokens_info

    def _sentences_to_info(self, doc: spacy.language.Doc) -> List[SentenceInfo]:
        sentences_info: List[SentenceInfo] = []
        for sent_id, sent in enumerate(doc.sents):
            token_indices = list(range(sent.start, sent.end))
            sentences_info.append(
                SentenceInfo(
                    sent_id=sent_id,
                    text=sent.text,
                    start_char=sent.start_char,
                    end_char=sent.end_char,
                    token_indices=token_indices,
                )
            )
        return sentences_info

    def _map_person_name(self, ent: spacy.tokens.Span) -> List[EntityHint]:
        hints: List[EntityHint] = []
        name_tokens = [tok for tok in ent if tok.is_alpha]

        if not name_tokens:
            return hints

        first = name_tokens[0]
        hints.append(EntityHint(text=first.text, label="name", start_char=first.idx, end_char=first.idx + len(first)))

        if len(name_tokens) > 1:
            last = name_tokens[-1]
            hints.append(EntityHint(text=last.text, label="surname", start_char=last.idx, end_char=last.idx + len(last)))

        return hints

    def _map_spacy_entity(self, ent: spacy.tokens.Span) -> List[EntityHint]:
        label = ent.label_
        text_lower = ent.text.lower()

        if label in ALLOWED_LABELS:
            return [EntityHint(text=ent.text, label=label, start_char=ent.start_char, end_char=ent.end_char)]

        if label == "persName":
            return self._map_person_name(ent)

        if label in {"placeName", "geogName"}:
            return [EntityHint(text=ent.text, label="city", start_char=ent.start_char, end_char=ent.end_char)]

        if label == "orgName":
            mapped_label = "school-name" if any(key in text_lower for key in ORG_SCHOOL_KEYWORDS) else "company"
            return [EntityHint(text=ent.text, label=mapped_label, start_char=ent.start_char, end_char=ent.end_char)]

        return []

    def _entities_to_hints(self, doc: spacy.language.Doc) -> List[EntityHint]:
        entity_hints: List[EntityHint] = []
        for ent in doc.ents:
            entity_hints.extend(self._map_spacy_entity(ent))
        return entity_hints

    def _regex_entities(self, text: str) -> List[EntityHint]:
        regex_hints: List[EntityHint] = []
        for hint in regex_labeling(text):
            if hint.label not in ALLOWED_LABELS:
                continue
            regex_hints.append(
                EntityHint(
                    text=hint.text,
                    label=hint.label,
                    start_char=hint.start_char,
                    end_char=hint.end_char,
                )
            )
        return regex_hints

    def _merge_entities(self, hints: List[EntityHint]) -> List[EntityHint]:
        seen = set()
        merged: List[EntityHint] = []

        for ent in sorted(hints, key=lambda e: (e.start_char, -(e.end_char - e.start_char))):
            key = (ent.start_char, ent.end_char, ent.label)
            if key in seen or ent.label not in ALLOWED_LABELS:
                continue
            merged.append(ent)
            seen.add(key)

        return merged

    def _redact_text(self, text: str, entities: List[EntityHint]) -> str:
        if not entities:
            return text

        redacted_parts: List[str] = []
        cursor = 0
        last_end = 0

        for ent in sorted(entities, key=lambda e: (e.start_char, -(e.end_char - e.start_char))):
            if ent.start_char < last_end:
                continue

            redacted_parts.append(text[cursor:ent.start_char])
            redacted_parts.append(f"[{ent.label.upper()}]")
            cursor = ent.end_char
            last_end = ent.end_char

        redacted_parts.append(text[cursor:])
        return "".join(redacted_parts)

    def __call__(self, text: str) -> PreprocessResult:
        doc = self.nlp(text)

        tokens = self._tokens_to_info(doc)
        sentences = self._sentences_to_info(doc)
        ner_entities = self._entities_to_hints(doc) if self.use_ner_hints else []
        regex_entities = self._regex_entities(text) if self.use_regex_hints else []

        merged_entities = self._merge_entities([*ner_entities, *regex_entities])

        redacted_text = self._redact_text(text, merged_entities)

        meta = {
            "use_ner_hints": self.use_ner_hints,
            "use_regex_hints": self.use_regex_hints,
            "num_tokens": len(tokens),
            "num_sentences": len(sentences),
            "num_entities_ner": len(ner_entities),
            "num_entities_regex": len(regex_entities),
            "num_entities": len(merged_entities),
        }

        return PreprocessResult(
            raw_text=text,
            tokens=tokens,
            sentences=sentences,
            entities=merged_entities,
            redacted_text=redacted_text,
            meta=meta,
        )
