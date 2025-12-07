import spacy

from spacy.tokens import Span


def shrink_spans(doc: spacy.language.Doc, label: str) -> spacy.language.Doc:
    new_ents = []
    for ent in doc.ents:
        if ent.label_ == label:
            start = ent.end - 1
            end = ent.end
            new_ents.append(Span(doc, start, end, label=ent.label))
        else:
            new_ents.append(ent)

    doc.ents = tuple(new_ents)

    return doc
