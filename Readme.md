# System Anonimizacji Tekstu

System do automatycznej detekcji i anonimizacji danych wrażliwych w polskich tekstach konwersacyjnych oparty
na bibliotece spaCy (Entity Rulers + Regex). System zastępuje wykryte informacje wrażliwe odpowiednimi etykietami
(np. {imię}, {pesel}) z zachowaniem świadomości kontekstu.

## Funkcje

- Automatyczne wykrywanie kategorii danych wrażliwych
- Klasyfikacja uwzględniająca kontekst
- Obsługa tekstów nieformalnych i dialogów
- Rozróżnianie podobnych typów danych na podstawie kontekstu
- Opcjonalny moduł generowania danych syntetycznych

## Instalacja

1. Zainstaluj wymagane zależności: `pip install -r requirements.txt`
2. Zainstaluj model spaCy: `python -m spacy download pl_core_news_lg`

## Użycie

1. Anonimizuj dowolny plik tekstowy z CLI:  
   `python -m labeling.cli path/do/tekstu.txt -o wynik.txt`
2. Domyślnie używany jest model `pl_core_news_md`. Możesz go zmienić, np.:  
   `python -m labeling.cli input.txt -o wynik.txt --model pl_core_news_lg`
3. Aby zobaczyć dostępne opcje: `python -m labeling.cli --help`
