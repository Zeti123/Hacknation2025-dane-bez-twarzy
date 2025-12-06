import spacy
from spacy import Language
from spacy.tokens import Span

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
    new_ents = []
    for ent in doc.ents:
        if ent.label_ == RELIGION_LABEL:
            start = ent.end - 1
            end = ent.end
            new_ents.append(Span(doc, start, end, label=ent.label))
        else:
            new_ents.append(ent)
    doc.ents = tuple(new_ents)
    return doc

def add_religion_entity_ruler(nlp: spacy.Language):
    ruler = nlp.add_pipe("entity_ruler", before="ner")

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