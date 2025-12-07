import spacy
from spacy import Language

from ._utils import shrink_spans

RELATIVE_LABEL = "relative"
RELATIVE_WORDS = [
    "matka", "ojciec",
    "mama", "tata",
    "syn", "córka",
    "brat", "siostra",
    "dziadek", "babcia",
    "wnuk", "wnuczka",
    "wujek", "ciocia",
    "kuzyn", "kuzynka",
    "teść", "teściowa",
    "zięć", "synowa",
    "szwagier", "szwagierka",
    "prababcia", "pradziadek",
    "prawnuk", "prawnuczka",
    "siostrzeniec", "siostrzenica",
    "bratanek", "bratanica",
    "macocha", "ojczym",
    "pasierb", "pasierbica",
    "małżonek", "małżonka",
    "mąż", "żona",
    "rodzic", "rodzice",
    "dziecko", "dzieci",
    "rodzeństwo", "ród",
    "krewny", "krewna",
    "powinowaty", "powinowata"
]


@Language.component("shrink_relative_spans")
def shrink_relative_spans(doc):
    return shrink_spans(doc, RELATIVE_LABEL)


def add_relative_entity_ruler(nlp: spacy.Language):
    ruler = nlp.add_pipe("entity_ruler", name="relative_ruler", before="ner")

    ruler.add_patterns([
        {
            "label": RELATIVE_LABEL,
            "pattern": [
                {"LEMMA": {"IN": RELATIVE_WORDS}}
            ],
        }
    ])

    ruler.add_patterns([
        {
            "label": RELATIVE_LABEL,
            "pattern": [
                {"LEMMA": "być"},
                {"POS": "ADJ", "OP": "?"},
                {"LEMMA": {"IN": RELATIVE_WORDS}},
            ],
        },
        {
            "label": RELATIVE_LABEL,
            "pattern": [
                {"LEMMA": "być"},
                {"POS": "NOUN", "OP": "?"},
                {"POS": "ADJ", "OP": "?"},
                {"LEMMA": {"IN": RELATIVE_WORDS}},
            ],
        },
    ])

    nlp.add_pipe("shrink_relative_spans", last=True)

    return nlp
