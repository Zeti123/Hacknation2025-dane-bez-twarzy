from typing import Dict, Tuple

PLACEHOLDER_MAP = {
    "name": "[name]",
    "surname": "[surname]",
    "age": "[age]",
    "date-of-birth": "[date-of-birth]",
    "date": "[date]",
    "sex": "[sex]",
    "religion": "[religion]",
    "political-view": "[political-view]",
    "ethnicity": "[ethnicity]",
    "sexual-orientation": "[sexual-orientation]",
    "health": "[health]",
    "relative": "[relative]",
    "city": "[city]",
    "address": "[address]",
    "email": "[email]",
    "phone": "[phone]",
    "pesel": "[pesel]",
    "document-number": "[document-number]",
    "company": "[company]",
    "school-name": "[school-name]",
    "job-title": "[job-title]",
    "bank-account": "[bank-account]",
    "credit-card-number": "[credit-card-number]",
    "username": "[username]",
    "secret": "[secret]",
}

LEGACY_HINT_TO_LABEL = {
    "persName": "name",
    "placeName": "city",
    "geogName": "address",
}


def label_to_placeholder(label: str) -> str:
    return PLACEHOLDER_MAP.get(label, "[unknown]")


def normalize_hint_label(label: str) -> str:
    return LEGACY_HINT_TO_LABEL.get(label, label)


def apply_placeholders(text: str, entities: Dict[Tuple[int, int], str]) -> str:
    spans = sorted(entities.items(), key=lambda kv: kv[0][0], reverse=True)
    out = text
    for (start, end), label in spans:
        out = out[:start] + label_to_placeholder(label) + out[end:]
    return out
