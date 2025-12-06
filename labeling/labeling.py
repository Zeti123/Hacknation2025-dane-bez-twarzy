import spacy

from pipes.religion import add_religion_entity_ruler
from preprocessor import SpacyPreprocessor


if __name__ == '__main__':
    nlp = spacy.load("pl_core_news_md")

    nlp = add_religion_entity_ruler(nlp)

    example_text = """
        "Reprezentujemy konsorcjum DataSafe, które zajmuje się anonimizacją dokumentów. "
        "Siedziba firmy znajduje się we Wrocławiu przy ulicy Kościuszki 10. "
        "Dane osobowe takich osób jak Jan Kowalski czy Anna Nowak muszą zostać zanonimizowane. "
        "Mój nr telefonu to 123-456-789. Nazywam się Krawiec i urodziłem się 20-10-2024."
        "Jestem zapalonym katolikiem, żona muzułmanką a mój brat to żyd"
    """

    preprocessor = SpacyPreprocessor(nlp)
    result = preprocessor(example_text)

    print("--- ENTITIES ---")
    for e in result.entities:
        print(e.label, "->", e.text)