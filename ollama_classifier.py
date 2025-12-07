import json
from typing import Dict, Iterator, List, Tuple

import ollama

try:
    from tqdm import tqdm
except ImportError:  # fallback if tqdm is not installed
    tqdm = None

from classes import ALLOWED_LABELS, Chunk, EntityHint


class OllamaEntityClassifier:
    """
    Klasyfikator encji korzystający z biblioteki `ollama` (bezpośrednio, bez ręcznego HTTP).
    Przyjmuje chunki z ContextManagera i zwraca mapowanie globalnych pozycji -> etykieta.
    """

    def __init__(
        self,
        model: str = "SpeakLeash/bielik-7b-instruct-v0.1-gguf:Q4_K_S",
        base_url: str = "http://localhost:11434",
        max_hints_per_call: int = 40,
        timeout: int = 360,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_hints_per_call = max_hints_per_call
        self.timeout = timeout
        # przekazujemy timeout do httpx.Client używanego przez bibliotekę
        self.client = ollama.Client(host=self.base_url, timeout=self.timeout)

    def _build_user_prompt(self, chunk: Chunk, hints_subset: List[EntityHint]) -> Tuple[str, Dict[int, EntityHint]]:
        local_id_to_hint: Dict[int, EntityHint] = {}
        candidates = []

        for local_id, h in enumerate(hints_subset):
            local_id_to_hint[local_id] = h
            candidates.append(
                {
                    "id": local_id,
                    "text": h.text,
                    "hint_label": h.label,
                    "start": h.start_char - chunk.start_char,
                    "end": h.end_char - chunk.start_char,
                }
            )

        candidates_json = json.dumps(candidates, ensure_ascii=False, indent=2)
        labels_text = "\n".join(f"- {label}" for label in ALLOWED_LABELS)

        user_prompt = f"""
Tekst dokumentu (fragment):

\"\"\"{chunk.text}\"\"\"

Lista kandydatów do klasyfikacji (id, fragment, wstępna etykieta, pozycja w TEKŚCIE FRAGMENTU):
{candidates_json}

Dla KAŻDEGO kandydata wybierz JEDNĄ etykietę z listy (lub "none" jeśli nie pasuje):

{labels_text}

Zwróć WYŁĄCZNIE JSON w formacie:

{{
  "entities": [
    {{"id": <liczba>, "label": "<etykieta>"}},
    ...
  ]
}}
"""
        return user_prompt, local_id_to_hint

    def _call_ollama(self, system_prompt: str, user_prompt: str) -> dict:
        try:
            resp = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                format="json",
                options={
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": 4016,
                },
            )
        except Exception as e:
            raise RuntimeError(f"Ollama chat failed (host={self.base_url}, model={self.model})") from e

        # Ollama chat zwraca: {"message": {"role": "...", "content": "..."} , ...}
        message = resp.get("message", {})
        content = message.get("content")
        if not content:
            raise ValueError(f"Ollama zwróciła niekompletną odpowiedź: {resp}")
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Ollama zwróciła nie-JSON: {content}") from e

    def classify_chunk(self, chunk: Chunk, verbose: bool = False) -> Dict[Tuple[int, int], str]:
        """
        Zwraca mapę: (start_char, end_char) -> label (łącznie z 'none' dla odrzuconych hintów).
        Hinty są wysyłane w paczkach pogrupowanych po wstępnych etykietach (tematycznie),
        aby uniknąć mieszania bardzo różnych kategorii w jednym zapytaniu.
        """
        result: Dict[Tuple[int, int], str] = {}
        ents = chunk.entities

        if not ents:
            return result

        # grupujemy po wstępnej etykiecie, a dopiero potem robimy batching
        grouped: Dict[str, List[EntityHint]] = {}
        for h in ents:
            grouped.setdefault(h.label or "unknown", []).append(h)

        for label_key, group in grouped.items():
            for batch_idx, i in enumerate(range(0, len(group), self.max_hints_per_call), start=1):
                subset = group[i : i + self.max_hints_per_call]

                user_prompt, local_id_to_hint = self._build_user_prompt(chunk, subset)
                if verbose:
                    print(
                        f"[ollama] label={label_key}, batch={batch_idx}, "
                        f"hints={len(subset)}, chunk_span=({chunk.start_char},{chunk.end_char})"
                    )
                raw = self._call_ollama(ENTITY_SYSTEM_PROMPT, user_prompt)

                # spodziewamy się: {"entities": [{"id": 0, "label": "name"}, ...]}
                for ent_info in raw.get("entities", []):
                    local_id = ent_info.get("id")
                    label = ent_info.get("label")
                    if local_id is None or label is None:
                        continue

                    hint = local_id_to_hint.get(local_id)
                    if not hint:
                        continue

                    key = (hint.start_char, hint.end_char)
                    result[key] = label

        return result

    def classify_document(
        self, chunks: List[Chunk], show_progress: bool = True, verbose: bool = False
    ) -> Dict[Tuple[int, int], str]:
        """
        Zwraca globalną mapę: (start_char, end_char) -> label
        dla całego dokumentu (wszystkie chunki).
        """
        global_result: Dict[Tuple[int, int], str] = {}
        for _, partial in self.classify_document_stream(
            chunks, show_progress=show_progress, verbose=verbose
        ):
            for key, label in partial.items():
                global_result[key] = label
        return global_result

    def classify_document_stream(
        self, chunks: List[Chunk], show_progress: bool = True, verbose: bool = False
    ) -> Iterator[Tuple[int, Dict[Tuple[int, int], str]]]:
        """
        Jak classify_document, ale zwraca wyniki na bieżąco per chunk.
        """
        iterator = enumerate(chunks)
        if show_progress and tqdm:
            iterator = enumerate(tqdm(chunks, desc="Ollama classify", unit="chunk"))

        for idx, chunk in iterator:
            partial = self.classify_chunk(chunk, verbose=verbose)
            yield idx, partial
    

ENTITY_SYSTEM_PROMPT = """
Jesteś asystentem ds. klasyfikacji danych osobowych.

Zadanie: dla każdego podanego kandydata (id, fragment tekstu) wybierz DOKŁADNIE jedną etykietę z listy lub "none", jeśli fragment nie jest daną osobową/wrażliwą.
Nigdy nie wymyślaj nowych etykiet i nie zmieniaj tekstu kandydatów. Skup się na danych osobowych i wrażliwych.

Dostępne etykiety:
1) Dane identyfikacyjne: name, surname, age, date-of-birth, date, sex, religion, political-view, ethnicity, sexual-orientation, health, relative
2) Lokalizacja/kontakt: city, address, email, phone
3) Dokumenty: pesel, document-number
4) Zawód/edukacja: company, school-name, job-title
5) Finanse: bank-account, credit-card-number
6) Loginy/sekrety: username, secret
7) none – jeśli fragment NIE pasuje do żadnej z powyższych etykiet

Format odpowiedzi (JSON-only):
{
  "entities": [
    {"id": <liczba>, "label": "<etykieta_z_listy_lub_none>"},
    ...
  ]
}
"""
