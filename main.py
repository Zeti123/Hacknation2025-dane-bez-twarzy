from labeling.anonymizer import anonymize

if __name__ == '__main__':
    with open("labeling/test_data.txt", "rt", encoding="utf-8") as file:
        text = file.read()
        result = anonymize(text)
        with open("output_broclaw.txt", "wt", encoding="utf-8") as out_file:
            out_file.write(result)