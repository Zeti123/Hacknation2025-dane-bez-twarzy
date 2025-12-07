import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple


@dataclass
class EntityHint:
    text: str
    label: str
    start_char: int
    end_char: int

    def __hash__(self):
        return hash((self.text, self.label, self.start_char, self.end_char))


def check_pesel(pesel: str) -> bool:
    if not re.fullmatch(r"\d{11}", pesel):
        return False

    weights = [1, 3, 7, 9, 1, 3, 7, 9, 1, 3]
    digits = list(map(int, pesel))
    checksum = sum(w * d for w, d in zip(weights, digits))
    control_digit = (10 - (checksum % 10)) % 10

    return control_digit == digits[-1]


def luhn_check(number: str) -> bool:
    """Validate number using the Luhn algorithm."""
    cleaned = normalize_digits(number)
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


def normalize_digits(value: str) -> str:
    replacements = str.maketrans({
        "o": "0", "O": "0",
        "l": "1", "I": "1",
        "B": "8", "S": "5", "Z": "2",
        "q": "9", "G": "6", "b": "6",
    })
    value = value.translate(replacements)
    return re.sub(r"\D", "", value)


def is_valid_phone(raw: str) -> bool:
    digits = normalize_digits(raw)
    if digits.startswith("48") and len(digits) == 11:
        digits = digits[2:]
    return len(digits) == 9


def is_valid_bank_account(raw: str) -> bool:
    cleaned = re.sub(r"(?i)^pl", "", raw.replace(" ", "").replace("-", ""))
    digits = normalize_digits(cleaned)
    return len(digits) == 26


def _pattern_defs() -> List[Tuple[str, re.Pattern, Optional[Callable[[str], bool]], int]]:
    return [
        ("pesel", re.compile(r"\b\d{11}\b"), check_pesel, 0),
        ("credit-card-number", re.compile(r"\b(?:\d[ -]?){13,19}\b"), luhn_check, 0),
        ("bank-account", re.compile(r"\b(?:PL\s*)?(?:\d[ -]?){26}\b", re.IGNORECASE), is_valid_bank_account, 0),
        ("document-number", re.compile(r"\b[A-Z]{2,3}\d{4,9}\b")),
        ("document-number", re.compile(r"\b\d{4}-\d{4}-\d{4}-\d{4}\b")),
        ("document-number", re.compile(r"\b\d{4}-\d{4}-\d{4}\b")),
        ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", re.IGNORECASE)),
        (
            "phone",
            re.compile(
                r"\b(?:\+?48[\s-]?)?(?:[\doOIlBGSq]{3}[\s-]?[\doOIlBGSq]{3}[\s-]?[\doOIlBGSq]{3})\b",
                re.IGNORECASE,
            ),
            is_valid_phone,
            0,
        ),
        (
            "date-of-birth",
            re.compile(
                r"(?i)(?:urodzony|urodzona|ur\.?|data urodzenia)[:\s-]*([0-3]?\d[./-][01]?\d[./-](?:\d{2}|\d{4}))"
            ),
            None,
            1,
        ),
        (
            "date-of-birth",
            re.compile(r"(?i)(?:urodzony|urodzona|ur\.?|data urodzenia)[:\s-]*(\d{4}-\d{2}-\d{2})"),
            None,
            1,
        ),
        (
            "school-name",
            re.compile(
                r"(?i)\b(?:szkoła|liceum|technikum|uniwersytet|akademia|politechnika)[^\n\r,.;]{0,50}"
            ),
            None,
            0,
        ),
        (
            "username",
            re.compile(r"(?i)(?:login|użytkownik|username)[:\s-]*([\w.@+-]{3,})"),
            None,
            1,
        ),
        (
            "secret",
            re.compile(r"(?i)(?:hasło|password|token|klucz api|api key|sekret)[:\s-]*([\w!@#$%^&*()\-_=+]{4,})"),
            None,
            1,
        ),
    ]


def regex_labeling(text: str) -> List[EntityHint]:
    hints: List[EntityHint] = []

    for label, pattern, *rest in _pattern_defs():
        validator: Optional[Callable[[str], bool]] = None
        group_idx = 0

        if rest:
            if len(rest) == 1:
                validator = rest[0]
            elif len(rest) == 2:
                validator, group_idx = rest

        for match in pattern.finditer(text):
            matched_text = match.group(group_idx)
            start, end = match.span(group_idx)

            if validator and not validator(matched_text):
                continue

            hints.append(EntityHint(
                text=matched_text,
                label=label,
                start_char=start,
                end_char=end
            ))

    return hints
