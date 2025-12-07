import argparse
from pathlib import Path
from typing import Sequence

from labeling.anonymizer import anonymize, DEFAULT_MODEL, DEFAULT_MAX_LEN


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Anonymize a plaintext file using spaCy + rule-based detectors.")
    parser.add_argument("input", type=Path, help="Path to the input text file.")
    parser.add_argument("-o", "--output", required=True, type=Path, help="Where to write the anonymized text.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"spaCy model to load (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LEN,
        help=f"Override spaCy max_length (default: {DEFAULT_MAX_LEN}).",
    )
    parser.add_argument(
        "--no-ner-hints",
        action="store_true",
        help="Disable spaCy NER hints (only rule-based entity rulers).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress timing/log output.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    text = args.input.read_text(encoding="utf-8")
    redacted = anonymize(
        text,
        model=args.model,
        max_length=args.max_length,
        use_ner_hints=not args.no_ner_hints,
        verbose=not args.quiet,
    )
    args.output.write_text(redacted, encoding="utf-8")
    if not args.quiet:
        print(f"Anonymized text written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
