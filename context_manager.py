
from typing import List, Optional, Tuple

from classes import Chunk, EntityHint, PreprocessResult, SentenceSpan, estimate_tokens_from_chars
from format import normalize_hint_label

MAX_CONTEXT_TOKENS = 8192
SYSTEM_TOKENS = 300          # na oko, w zależności od tego jak długi masz system prompt
RESPONSE_TOKENS = 512        # górny limit na JSON
SAFETY_MARGIN = 0.7          # zostaw trochę luzu

MAX_TEXT_TOKENS = int((MAX_CONTEXT_TOKENS - SYSTEM_TOKENS - RESPONSE_TOKENS) * SAFETY_MARGIN)


class ContextManager:
    """
    Buduje poręczne chunki tekstu (ciągłe fragmenty zdań) tak,
    aby zmieścić się w kontekście modelu i nie gubić NER hintów.
    """

    def __init__(self, max_text_tokens: int = MAX_TEXT_TOKENS) -> None:
        self.max_text_tokens = max_text_tokens

    def _sentence_spans(self, pre: PreprocessResult) -> List[SentenceSpan]:
        spans: List[SentenceSpan] = []
        for sent in pre.sentences:
            sent_text = pre.raw_text[sent.start_char : sent.end_char]
            spans.append(
                SentenceSpan(
                    text=sent_text,
                    start_char=sent.start_char,
                    end_char=sent.end_char,
                    est_tokens=estimate_tokens_from_chars(sent_text),
                )
            )
        return spans

    def _slice_entities(
        self, entities: List[EntityHint], chunk_start: int, chunk_end: int, start_idx: int
    ) -> Tuple[List[EntityHint], int]:
        """
        Zwraca encje mieszczące się w [chunk_start, chunk_end) oraz indeks,
        od którego zaczyna się kolejny chunk (iterujemy jednokrotnie po liście).
        """
        collected: List[EntityHint] = []
        idx = start_idx
        n = len(entities)

        # pomijamy encje, które kończą się przed chunk_start
        while idx < n and entities[idx].end_char <= chunk_start:
            idx += 1

        # zbieramy encje, które mieszczą się w chunku
        while idx < n:
            ent = entities[idx]
            if ent.start_char >= chunk_end:
                break
            if ent.start_char >= chunk_start and ent.end_char <= chunk_end:
                collected.append(ent)
            idx += 1

        return collected, idx

    def chunk(self, pre: PreprocessResult) -> List[Chunk]:
        """
        Chunk:
        - text – kawałek tekstu (1..N zdań),
        - start_char, end_char – w oryginalnym tekście,
        - sentences – lista zdań w tym kawałku,
        - entities – lista EntityHint w tym kawałku.
        """
        spans = self._sentence_spans(pre)

        # jeśli brak segmentacji zdań, bierzemy cały tekst w jednym kawałku
        if not spans:
            return [
                Chunk(
                    text=pre.raw_text,
                    start_char=0,
                    end_char=len(pre.raw_text),
                    sentences=[],
                    entities=pre.entities,
                )
            ]

        entities = sorted(pre.entities, key=lambda e: e.start_char)
        chunks: List[Chunk] = []
        current_spans: List[SentenceSpan] = []
        current_tokens = 0
        chunk_start = 0
        ent_idx = 0

        for span in spans:
            if current_spans and current_tokens + span.est_tokens > self.max_text_tokens:
                chunk_end = current_spans[-1].end_char
                chunk_text = pre.raw_text[chunk_start:chunk_end]
                chunk_entities, ent_idx = self._slice_entities(entities, chunk_start, chunk_end, ent_idx)
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        start_char=chunk_start,
                        end_char=chunk_end,
                        sentences=current_spans,
                        entities=chunk_entities,
                    )
                )
                current_spans = []
                current_tokens = 0

            if not current_spans:
                chunk_start = span.start_char

            current_spans.append(span)
            current_tokens += span.est_tokens

        if current_spans:
            chunk_end = current_spans[-1].end_char
            chunk_text = pre.raw_text[chunk_start:chunk_end]
            chunk_entities, _ = self._slice_entities(entities, chunk_start, chunk_end, ent_idx)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    start_char=chunk_start,
                    end_char=chunk_end,
                    sentences=current_spans,
                    entities=chunk_entities,
                )
            )

        return chunks

    def chunk_by_hints(
        self,
        pre: PreprocessResult,
        hints: Optional[List[EntityHint]] = None,
        sentence_radius: int = 0,
        normalize_labels: bool = True,
    ) -> List[Chunk]:
        """
        Tworzy pojedynczy chunk per hint: zawiera zdanie z hintem (+ew. sąsiednie zdania).
        Użyteczne do walidacji / filtrowania hintów bez ryzyka rozwodnienia kontekstu.
        """
        spans = self._sentence_spans(pre)
        if not spans:
            return []

        selected_hints = sorted(hints if hints is not None else pre.entities, key=lambda e: e.start_char)
        if not selected_hints:
            return []

        chunks: List[Chunk] = []

        for hint in selected_hints:
            # znajdź zdanie, w którym zaczyna się hint
            sent_idx = None
            for i, s in enumerate(spans):
                if s.start_char <= hint.start_char < s.end_char:
                    sent_idx = i
                    break
            if sent_idx is None:
                continue

            lo = max(0, sent_idx - sentence_radius)
            hi = min(len(spans) - 1, sent_idx + sentence_radius)
            picked = spans[lo : hi + 1]

            chunk_start = picked[0].start_char
            chunk_end = picked[-1].end_char
            chunk_text = pre.raw_text[chunk_start:chunk_end]

            label = normalize_hint_label(hint.label) if normalize_labels else hint.label
            norm_hint = EntityHint(
                text=hint.text,
                label=label,
                start_char=hint.start_char,
                end_char=hint.end_char,
            )

            chunks.append(
                Chunk(
                    text=chunk_text,
                    start_char=chunk_start,
                    end_char=chunk_end,
                    sentences=picked,
                    entities=[norm_hint],
                )
            )

        return chunks
