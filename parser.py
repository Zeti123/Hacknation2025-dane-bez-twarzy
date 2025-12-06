import spacy
from dataclasses import dataclass
from typing import List, Optional, Any, Dict

try:
    nlp = spacy.load("pl_core_news_md")
except OSError as e:
    raise RuntimeError(
        "Polish model 'pl_core_news_md' is not installed. "
        "Run: python -m spacy download pl_core_news_md"
    ) from e

nlp

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
    meta: Dict[str, Any]

class SpacyPreprocessor:
    def __init__(
        self,
        model_name: str = "pl_core_news_md",
        use_ner_hints: bool = True,
        disable: Optional[List[str]] = None,
    ) -> None:
        """
        Wrapper around spaCy Polish pipeline.

        Parameters
        ----------
        model_name : str
            Name of the spaCy model to load.
        use_ner_hints : bool
            Whether to extract NER hints from spaCy doc.ents.
        disable : list[str] | None
            Optional list of pipeline components to disable for speed.
            Example: ["tagger", "parser", "attribute_ruler", "lemmatizer"]
        """
        self.model_name = model_name
        self.use_ner_hints = use_ner_hints
        self.disable = disable or []

        # Load or reuse existing global nlp if possible
        try:
            self.nlp = spacy.load(model_name, disable=self.disable)
        except OSError as e:
            raise RuntimeError(
                f"spaCy model '{model_name}' is not installed. "
                f"Install with: python -m spacy download {model_name}"
            ) from e

    def _tokens_to_info(self, doc: "spacy.tokens.Doc") -> List[TokenInfo]:
        tokens_info: List[TokenInfo] = []
        for i, token in enumerate(doc):
            tokens_info.append(
                TokenInfo(
                    idx=i,
                    text=token.text,
                    lemma=token.lemma_,
                    pos=token.pos_,
                    tag=token.tag_,
                    morph=token.morph,
                    dep=token.dep_,
                    head=token.head.i,
                    is_stop=token.is_stop,
                    is_punct=token.is_punct,
                    whitespace=token.whitespace_,
                )
            )
        return tokens_info

    def _sentences_to_info(self, doc: "spacy.tokens.Doc") -> List[SentenceInfo]:
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

    def _entities_to_hints(self, doc: "spacy.tokens.Doc") -> List[EntityHint]:
        entity_hints: List[EntityHint] = []
        for ent in doc.ents:
            entity_hints.append(
                EntityHint(
                    text=ent.text,
                    label=ent.label_,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                )
            )
        return entity_hints

    def __call__(self, text: str) -> PreprocessResult:
        """
        Run full preprocessing pipeline on raw text.
        """
        doc = self.nlp(text)

        tokens = self._tokens_to_info(doc)
        sentences = self._sentences_to_info(doc)
        entities = self._entities_to_hints(doc) if self.use_ner_hints else []

        meta = {
            "model_name": self.model_name,
            "use_ner_hints": self.use_ner_hints,
            "num_tokens": len(tokens),
            "num_sentences": len(sentences),
            "num_entities": len(entities),
        }

        return PreprocessResult(
            raw_text=text,
            tokens=tokens,
            sentences=sentences,
            entities=entities,
            meta=meta,
        )