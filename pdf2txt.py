from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


def extract_text_from_pdf(pdf_path: Path) -> str:
    if PdfReader is None:
        raise RuntimeError(
            "Missing dependency 'pypdf'. Install it with: pip install pypdf"
        )

    reader = PdfReader(str(pdf_path))
    page_texts: list[str] = []

    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text and not text.endswith("\n"):
            text += "\n"
        page_texts.append(text)

    return "".join(page_texts)


def convert_pdfs(input_dir: Path, output_dir: Path, recursive: bool = False) -> tuple[int, int]:
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist or is not a directory: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    pattern = "**/*.pdf" if recursive else "*.pdf"
    pdf_files = sorted(input_dir.glob(pattern))

    converted = 0
    failed = 0

    for pdf_path in pdf_files:
        try:
            relative = pdf_path.relative_to(input_dir)
            target_path = (output_dir / relative).with_suffix(".txt")
            target_path.parent.mkdir(parents=True, exist_ok=True)

            text = extract_text_from_pdf(pdf_path)
            target_path.write_text(text, encoding="utf-8")
            print(f"Converted: {pdf_path} -> {target_path}")
            converted += 1
        except Exception as error:  # noqa: BLE001
            print(f"Failed: {pdf_path} ({error})", file=sys.stderr)
            failed += 1

    return converted, failed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan a directory for PDF files and convert each one to a TXT file."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing PDF files.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory where TXT files will be written.",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Scan subdirectories recursively.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        converted, failed = convert_pdfs(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            recursive=args.recursive,
        )
    except Exception as error:  # noqa: BLE001
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Done. Converted: {converted}, Failed: {failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
