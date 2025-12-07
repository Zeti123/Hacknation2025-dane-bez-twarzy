import spacy
from spacy.language import Language
from spacy.tokens import Span

DATE_LABEL = "date"
POLISH_MONTHS = [
    "stycznia", "lutego", "marca", "kwietnia", "maja", "czerwca",
    "lipca", "sierpnia", "września", "października", "listopada", "grudnia"
]

@Language.component("shrink_date_spans")
def shrink_date_spans(doc):
    """Opcjonalnie skraca spany daty do ostatniego tokenu, np. tylko rok."""
    new_ents = []
    for ent in doc.ents:
        if ent.label_ == DATE_LABEL:
            new_ents.append(ent)  # tutaj możesz zmodyfikować, jeśli chcesz skracać
        else:
            new_ents.append(ent)
    doc.ents = tuple(new_ents)
    return doc

def add_date_entity_ruler(nlp: spacy.Language):
    ruler = nlp.add_pipe("entity_ruler", name="date_ruler", before="ner")

    # formaty cyfrowe: 01.02.2020, 1/2/2020
    ruler.add_patterns([
        {
            "label": DATE_LABEL,
            "pattern": [
                {"SHAPE": "dd"}, {"TEXT": {"REGEX": "[./-]"}}, {"SHAPE": "dd"}, {"TEXT": {"REGEX": "[./-]"}}, {"SHAPE": "dddd"}
            ]
        },
        {
            "label": DATE_LABEL,
            "pattern": [
                {"SHAPE": "d"}, {"TEXT": {"REGEX": "[./-]"}}, {"SHAPE": "d"}, {"TEXT": {"REGEX": "[./-]"}}, {"SHAPE": "dddd"}
            ]
        },
    ])

    # formaty słowne: 1 stycznia 2020, 1 stycznia 2020 r.
    ruler.add_patterns([
        {
            "label": DATE_LABEL,
            "pattern": [
                {"SHAPE": "d"},  # dzień
                {"LOWER": {"IN": POLISH_MONTHS}},  # miesiąc
                {"SHAPE": "dddd"},  # rok
                {"LOWER": "r", "OP": "?"},  # opcjonalne "r."
            ]
        }
    ])

    nlp.add_pipe("shrink_date_spans", last=True)
    return nlp