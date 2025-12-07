import spacy
from spacy import Language

# Keyword-driven entity ruler for labels that can be picked up from single words.

KEYWORD_MAP = {
    "political-view": [
        "liberał", "liberalny", "konserwatysta", "konserwatywny",
        "socjalista", "lewicowiec", "prawicowiec", "centrowy",
        "anarchista", "narodowiec",
    ],
    "ethnicity": [
        "polak", "polka", "ukrainiec", "ukrainka", "niemiec", "niemka",
        "rosjanin", "rosjanka", "żyd", "rom", "romka", "białorusin",
    ],
    "sexual-orientation": [
        "gej", "lesbijka", "biseksualny", "biseksualna",
        "heteroseksualny", "heteroseksualna", "panseksualny",
        "panseksualna", "queer",
    ],
    "health": [
        "choroba", "chora", "chory", "depresja", "gorączka", "ból",
        "infekcja", "astma", "rak", "grypa", "schorzenie",
    ],
    "job-title": [
        "kierownik", "dyrektor", "prezes", "szef", "nauczyciel",
        "nauczycielka", "psycholog", "lekarz", "prawnik", "magazynier",
        "student", "profesor", "inżynier", "doktor",
    ],
    "school-name": [
        "szkoła", "szkole", "szkolny", "liceum", "technikum",
        "uniwersytet", "akademia", "politechnika",
    ],
}


def _keyword_patterns():
    patterns = []
    for label, keywords in KEYWORD_MAP.items():
        for kw in keywords:
            patterns.append({"label": label, "pattern": [{"LOWER": kw}]})
    return patterns


def add_keyword_entity_ruler(nlp: spacy.Language):
    ruler = nlp.add_pipe(
        "entity_ruler",
        name="keyword_ruler",
        after="ner",
        config={"overwrite_ents": True},
    )
    ruler.add_patterns(_keyword_patterns())
    return nlp
