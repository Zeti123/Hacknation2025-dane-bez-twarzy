import spacy
from spacy import Language
from spacy.tokens import Span

SEX_LABEL = "sex"
SEX_WORDS = [
    "mężczyzna",
    "kobieta",
]

@Language.component("shrink_sex_spans")
def shrink_sex_spans(doc):
    new_ents = []
    for ent in doc.ents:
        if ent.label_ == SEX_LABEL:
            start = ent.end - 1
            end = ent.end
            new_ents.append(Span(doc, start, end, label=ent.label))
        else:
            new_ents.append(ent)
    doc.ents = tuple(new_ents)
    return doc

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