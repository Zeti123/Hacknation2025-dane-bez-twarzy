"""
Rule-based entity detection implemented with spaCy's EntityRuler.

The goal is to replace ad-hoc regex post-processing with a pipeline-aware
component that benefits from spaCy's tokenization and span management.
Patterns are defined using token attributes (including regex on TEXT) and
validated where needed (e.g., PESEL checksum, Luhn for cards).
"""

import re
import spacy
from spacy.language import Language
from spacy.tokens import Span

RULE_SOURCE = "rule-based"


# --- Normalization helpers -------------------------------------------------

def _normalize_digits(value: str) -> str:
    """Convert common OCR/typo characters to digits and strip non-digits."""
    replacements = str.maketrans({
        "o": "0", "O": "0",
        "l": "1", "I": "1",
        "B": "8", "S": "5", "Z": "2",
        "q": "9", "G": "6", "b": "6",
    })
    value = value.translate(replacements)
    return re.sub(r"\D", "", value)


# --- Validators -------------------------------------------------------------

def check_pesel(pesel: str) -> bool:
    digits = _normalize_digits(pesel)
    if not re.fullmatch(r"\d{11}", digits):
        return False

    weights = [1, 3, 7, 9, 1, 3, 7, 9, 1, 3]
    checksum = sum(w * int(d) for w, d in zip(weights, digits))
    control_digit = (10 - (checksum % 10)) % 10
    return control_digit == int(digits[-1])


def luhn_check(number: str) -> bool:
    """Validate number using the Luhn algorithm."""
    cleaned = _normalize_digits(number)
    if not cleaned.isdigit() or len(cleaned) < 13:
        return False

    total = 0
    reverse_digits = cleaned[::-1]
    for i, d in enumerate(reverse_digits):
        n = int(d)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n

    return total % 10 == 0


def is_valid_phone(raw: str) -> bool:
    digits = _normalize_digits(raw)
    # Accept global phone lengths (after stripping country code/separators).
    return 7 <= len(digits) <= 9


def is_valid_bank_account(raw: str) -> bool:
    cleaned = re.sub(r"(?i)^pl", "", raw.replace(" ", "").replace("-", ""))
    digits = _normalize_digits(cleaned)
    return len(digits) == 26


VALIDATORS = {
    "pesel": check_pesel,
    "credit-card-number": luhn_check,
    "bank-account": is_valid_bank_account,
    "phone": is_valid_phone,
}


# --- Pattern helpers -------------------------------------------------------

ADDRESS_PREFIXES = ["ul.", "ul", "ulica", "al.", "al", "aleja", "pl.", "pl", "plac", "os.", "osiedle", "pi.", "pi"]
MONTHS = [
    "stycznia", "lutego", "marca", "kwietnia", "maja", "czerwca",
    "lipca", "sierpnia", "września", "października", "listopada", "grudnia",
]


def _patterns():
    patterns = []

    number_token = {"TEXT": {"REGEX": r"[0-9oOIlBGSq\-]{2,}"}}
    phone_token = {"LIKE_NUM": True}
    phone_hybrid = {"TEXT": {"REGEX": r"\+?\d[\d\-()]{2,}"}}

    # PESEL
    patterns.append({
        "label": "pesel",
        "id": RULE_SOURCE,
        "pattern": [{"TEXT": {"REGEX": r"\d{11}"}}],
    })

    # Credit cards and bank accounts (validated later)
    patterns.append({
        "label": "credit-card-number",
        "id": RULE_SOURCE,
        "pattern": [{**number_token, "OP": "+"}],
    })
    patterns.append({
        "label": "bank-account",
        "id": RULE_SOURCE,
        "pattern": [
            {"TEXT": {"REGEX": r"(?i)pl"}, "OP": "?"},
            {**number_token, "OP": "+"},
        ],
    })

    # Document numbers (two forms)
    patterns.append({
        "label": "document-number",
        "id": RULE_SOURCE,
        "pattern": [{"TEXT": {"REGEX": r"[A-Z]{2,3}\d{4,9}"}}],
    })
    patterns.append({
        "label": "document-number",
        "id": RULE_SOURCE,
        "pattern": [{"TEXT": {"REGEX": r"\d{4}-\d{4}-\d{4}(?:-\d{4})?"}}],
    })

    # Email (case-insensitive, supports + and multiple subdomains)
    patterns.append({
        "label": "email",
        "id": RULE_SOURCE,
        "pattern": [{"TEXT": {"REGEX": r"(?i)[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}"}}],
    })

    # Phone: allow global formats with separators/parentheses and country codes
    patterns.append({
        "label": "phone",
        "id": RULE_SOURCE,
        "pattern": [
            {"IS_PUNCT": True, "OP": "*"},
            {"TEXT": {"REGEX": r"\\+?\\d{1,3}"}, "OP": "?"},
            {"IS_PUNCT": True, "OP": "*"},
            {**phone_token, "OP": "+"},
            {"IS_PUNCT": True, "OP": "*"},
            {**phone_hybrid, "OP": "*"},
        ],
    })
    patterns.append({
        "label": "phone",
        "id": RULE_SOURCE,
        "pattern": [
            {"IS_PUNCT": True, "OP": "*"},
            phone_hybrid,
            {"IS_PUNCT": True, "OP": "*"},
        ],
    })

    # Date of birth phrases
    patterns.append({
        "label": "date-of-birth",
        "id": RULE_SOURCE,
        "pattern": [
            {"LOWER": {"IN": ["urodzony", "urodzona", "ur.", "ur", "data"]}},
            {"LOWER": "urodzenia", "OP": "?"},
            {"IS_PUNCT": True, "OP": "*"},
            {"TEXT": {"REGEX": r"[0-3]?\d[./-][01]?\d[./-](?:\d{2}|\d{4})"}},
        ],
    })
    patterns.append({
        "label": "date-of-birth",
        "id": RULE_SOURCE,
        "pattern": [
            {"LOWER": {"IN": ["urodzony", "urodzona", "ur.", "ur", "data"]}},
            {"LOWER": "urodzenia", "OP": "?"},
            {"IS_PUNCT": True, "OP": "*"},
            {"TEXT": {"REGEX": r"\d{4}-\d{2}-\d{2}"}},
        ],
    })

    # Generic date formats (numeric)
    patterns.append({
        "label": "date",
        "id": RULE_SOURCE,
        "pattern": [{"TEXT": {"REGEX": r"[0-3]?\d[./-][01]?\d[./-](?:\d{2}|\d{4})"}}],
    })

    # Generic date formats (word month)
    patterns.append({
        "label": "date",
        "id": RULE_SOURCE,
        "pattern": [
            {"TEXT": {"REGEX": r"[0-3]?\d"}},
            {"LOWER": {"IN": MONTHS}},
            {"TEXT": {"REGEX": r"\d{2,4}"}},
            {"LOWER": "r", "OP": "?"},
        ],
    })

    # Address
    patterns.append({
        "label": "address",
        "id": RULE_SOURCE,
        "pattern": [
            {"LOWER": {"IN": ADDRESS_PREFIXES}},
            {"IS_TITLE": True, "OP": "+"},
            {"TEXT": {"REGEX": r"\d[\w/]*"}},
            {"TEXT": {"REGEX": r"\d{2}-\d{3}"}, "OP": "?"},
            {"IS_TITLE": True, "OP": "*"},
        ],
    })

    # School name
    patterns.append({
        "label": "school-name",
        "id": RULE_SOURCE,
        "pattern": [
            {"LOWER": {"IN": ["szkoła", "liceum", "technikum", "uniwersytet", "akademia", "politechnika"]}},
            {"IS_PUNCT": True, "OP": "*"},
            {"IS_TITLE": True, "OP": "+"},
        ],
    })

    # Username/login
    patterns.append({
        "label": "username",
        "id": RULE_SOURCE,
        "pattern": [
            {"LOWER": {"IN": ["login", "username", "użytkownik"]}},
            {"IS_PUNCT": True, "OP": "*"},
            {"TEXT": {"REGEX": r"[\w.@+-]{3,}"}},
        ],
    })

    # Secrets / sensitive keys
    patterns.append({
        "label": "secret",
        "id": RULE_SOURCE,
        "pattern": [
            {"LOWER": {"IN": ["hasło", "password", "token", "sekret"]}},
            {"IS_PUNCT": True, "OP": "*"},
            {"TEXT": {"REGEX": r"[\w!@#$%^&*()\\-_=+]{4,}"}},
        ],
    })
    patterns.append({
        "label": "secret",
        "id": RULE_SOURCE,
        "pattern": [
            {"LOWER": "api"},
            {"LOWER": "key", "OP": "?"},
            {"IS_PUNCT": True, "OP": "*"},
            {"TEXT": {"REGEX": r"[\w!@#$%^&*()\\-_=+]{4,}"}},
        ],
    })

    return patterns


EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", re.IGNORECASE)
PHONE_RE = re.compile(r"\+?\d[\d\s\-()]{6,}\d")


@Language.component("regex_contact_entities")
def regex_contact_entities(doc):
    """Add email/phone spans detected via regex on raw text."""
    spans = []
    existing = list(doc.ents)

    def _overlaps(span):
        return [e for e in existing if span.start < e.end and e.start < span.end] + \
               [e for e in spans if span.start < e.end and e.start < span.end]
    for match in EMAIL_RE.finditer(doc.text):
        span = doc.char_span(match.start(), match.end(), label="email", kb_id=RULE_SOURCE, alignment_mode="contract")
        if span:
            overlaps = _overlaps(span)
            if overlaps:
                existing = [e for e in existing if e not in overlaps]
            spans.append(span)

    for match in PHONE_RE.finditer(doc.text):
        if not is_valid_phone(match.group()):
            continue
        span = doc.char_span(match.start(), match.end(), label="phone", kb_id=RULE_SOURCE, alignment_mode="contract")
        if span:
            overlaps = _overlaps(span)
            if overlaps:
                existing = [e for e in existing if e not in overlaps]
            spans.append(span)

    if spans:
        doc.ents = tuple(existing) + tuple(spans)
    return doc


@Language.component("filter_rule_spans")
def filter_rule_spans(doc):
    """Drop rule-based spans that fail validation; keep others untouched."""
    filtered = []
    for ent in doc.ents:
        if ent.ent_id_ != RULE_SOURCE and ent.kb_id_ != RULE_SOURCE:
            filtered.append(ent)
            continue

        validator = VALIDATORS.get(ent.label_)
        if validator and not validator(ent.text):
            continue
        filtered.append(ent)

    doc.ents = tuple(filtered)
    return doc


def add_rule_entity_ruler(nlp: spacy.Language):
    ruler = nlp.add_pipe(
        "entity_ruler",
        name="rule_entity_ruler",
        after="ner",
        config={"overwrite_ents": True},
    )
    ruler.add_patterns(_patterns())
    nlp.add_pipe("regex_contact_entities", after="rule_entity_ruler")
    nlp.add_pipe("filter_rule_spans", after="regex_contact_entities")
    return nlp
