import spacy
from spacy import Language

from ._utils import shrink_spans

SEX_LABEL = "sex"
SEX_WORDS = [
    "mężczyzna",
    "kobieta",
]


@Language.component("shrink_sex_spans")
def shrink_sex_spans(doc):
    return shrink_spans(doc, SEX_LABEL)


def add_sex_entity_ruler(nlp: spacy.Language):
    ruler = nlp.add_pipe("entity_ruler", name="sex_ruler", before="ner")

    ruler.add_patterns([
        {
            "label": SEX_LABEL,
            "pattern": [
                {"LEMMA": {"IN": SEX_WORDS}}
            ],
        }
    ])

    ruler.add_patterns([
        {
            "label": SEX_LABEL,
            "pattern": [
                {"LEMMA": "być"},
                {"POS": "ADJ", "OP": "?"},
                {"LEMMA": {"IN": SEX_WORDS}},
            ],
        },
    ])

    nlp.add_pipe("shrink_sex_spans", last=True)

    return nlp
