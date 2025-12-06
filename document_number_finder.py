import re
from classes import EntityHint
from typing import List


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
    number = number.replace(" ", "").replace("-", "")
    if not number.isdigit():
        return False

    total = 0
    reverse_digits = number[::-1]

    for i, d in enumerate(reverse_digits):
        n = int(d)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n

    return total % 10 == 0

def find_documents_numbers(text: str) -> List[EntityHint]:
    patterns = {
        "pesel": r"\b\d{11}\b",
        "dowod": r"\b[A-Z]{3}\d{6}\b",
        "paszport": r"\b[A-Z]{2}\d{7}\b",
        "karta": r"\b(?:\d[ -]?){13,19}\b",
    }

    hints: List[EntityHint] = []

    for label, pattern in patterns.items():
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            matched_text = match.group()

            # Dodatkowa walidacja PESEL
            if label == "pesel" and not check_pesel(matched_text):
                continue

            if label == "karta" and not luhn_check(matched_text):
                continue

            hints.append(EntityHint(
                text=matched_text,
                label=label,
                start_char=match.start(),
                end_char=match.end()
            ))

    return hints
