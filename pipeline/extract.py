"""
extract.py — Pull clean text out of corpus PDFs.

Flags files with suspiciously little extracted text (likely scans or
broken InPage text layers) so they can be routed to OCR instead.
"""

import json
import sys
from pathlib import Path

import pdfplumber

CORPUS_RAW = Path.home() / "Downloads/Projects/manar-prep/corpus/raw"
CORPUS_CLEAN = Path.home() / "Downloads/Projects/manar-prep/corpus/clean"

MIN_CHARS_PER_PAGE = 40


def extract_pdf(pdf_path: Path) -> dict:
    pages_text = []
    suspect_pages = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages_text.append(text)
            if len(text.strip()) < MIN_CHARS_PER_PAGE:
                suspect_pages.append(i + 1)

    full_text = "\n\n".join(pages_text)
    return {
        "source": pdf_path.name,
        "num_pages": len(pages_text),
        "suspect_pages": suspect_pages,
        "text": full_text,
    }


def main():
    CORPUS_CLEAN.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(CORPUS_RAW.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {CORPUS_RAW}")
        sys.exit(1)

    report_lines = []

    for pdf_path in pdf_files:
        print(f"Extracting: {pdf_path.name}")
        result = extract_pdf(pdf_path)

        out_path = CORPUS_CLEAN / (pdf_path.stem + ".txt")
        out_path.write_text(result["text"], encoding="utf-8")

        flag = ""
        if result["suspect_pages"]:
            pct = len(result["suspect_pages"]) / result["num_pages"] * 100
            flag = f"  ⚠️  {len(result['suspect_pages'])}/{result['num_pages']} pages ({pct:.0f}%) low-text — possible scan/broken layer"
            print(flag)

        report_lines.append({
            "source": pdf_path.name,
            "num_pages": result["num_pages"],
            "suspect_pages": result["suspect_pages"],
            "output": str(out_path.name),
        })

    report_path = CORPUS_CLEAN / "_extraction_report.json"
    report_path.write_text(json.dumps(report_lines, indent=2, ensure_ascii=False), encoding="utf-8")

    flagged = [r for r in report_lines if r["suspect_pages"]]
    print(f"\nDone. {len(pdf_files)} PDFs processed, {len(flagged)} flagged for review.")
    print(f"Report: {report_path}")
    if flagged:
        print("\nFiles needing OCR review:")
        for r in flagged:
            print(f"  - {r['source']} (pages: {r['suspect_pages']})")


if __name__ == "__main__":
    main()
