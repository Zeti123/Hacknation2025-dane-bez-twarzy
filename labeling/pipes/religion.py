import spacy
from spacy import Language

from ._utils import shrink_spans

RELIGION_LABEL = "religion"
RELIGION_WORDS = [
    "katolik",
    "katoliczka",
    "prawosławny",
    "prawosławna",
    "muzułmanin",
    "muzułmanka",
    "protestant",
    "żyd",
    "buddysta",
]


@Language.component("shrink_religion_spans")
def shrink_religion_spans(doc):
    return shrink_spans(doc, RELIGION_LABEL)


def add_religion_entity_ruler(nlp: spacy.Language):
    ruler = nlp.add_pipe(
        "entity_ruler",
        name="religion_ruler",
        after="ner",
        config={"overwrite_ents": True},
    )

    ruler.add_patterns([
        {
            "label": RELIGION_LABEL,
            "pattern": [
                {"LEMMA": {"IN": RELIGION_WORDS}}
            ],
        }
    ])

    ruler.add_patterns([
        {
            "label": RELIGION_LABEL,
            "pattern": [
                {"LEMMA": "być"},
                {"POS": "ADJ", "OP": "?"},
                {"LEMMA": {"IN": RELIGION_WORDS}},
            ],
        },
        {
            "label": RELIGION_LABEL,
            "pattern": [
                {"LEMMA": "być"},
                {"POS": "NOUN", "OP": "?"},
                {"POS": "ADJ", "OP": "?"},
                {"LEMMA": {"IN": RELIGION_WORDS}},
            ],
        },
    ])

    nlp.add_pipe("shrink_religion_spans", last=True)

    return nlp
