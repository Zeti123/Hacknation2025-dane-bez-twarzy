import spacy
from spacy import Language
from spacy.tokens import Span

AGE_LABEL = "age"

@Language.component("shrink_age_spans")
def shrink_age_spans(doc):
    new_ents = []
    for ent in doc.ents:
        if ent.label_ == AGE_LABEL:
            for token in ent:
                if token.pos_ == "NUM":
                    new_ent = Span(doc, token.i, token.i + 1, label=AGE_LABEL)
                    new_ents.append(new_ent)
                    break
        else:
            new_ents.append(ent)
    doc.ents = tuple(new_ents)
    return doc


def add_age_entity_ruler(nlp: spacy.Language):
    ruler = nlp.add_pipe("entity_ruler", name="age_ruler", before="ner")

    ruler.add_patterns([
        {
            "label": AGE_LABEL,
            "pattern": [
                {"POS": "NUM"},
                {"LEMMA": "rok"},
            ],
        }
    ])

    nlp.add_pipe("shrink_age_spans", last=True)

    return nlp
