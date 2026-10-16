import json
import os
import re

import pymupdf

from prodml import config

PDF_PATH = os.path.join(config.BASE_DIR, "data", "raw", "egyptian_civil_code.pdf")
JSON_PATH = os.path.join(config.BASE_DIR, "data", "raw", "civil_code_articles.json")

AR_HEADER = re.compile(r"(?<![\u0621-\u064A])مادة\s*\)?\s*\d+\s*\)?")

EN_HEADER = re.compile(r"^[ \t]*Article[ \t]+(\d+)[ \t]*$", re.MULTILINE)

AR_STOP = re.compile(r"\n(?=(?:الفصل|الباب|الكتاب|القسم)\s)")
EN_STOP = re.compile(
    r"\n(?=(?:SECTION|Section|CHAPTER|Chapter|BOOK|Book|FIRST PART|SECOND PART)\b)"
)

REPEALED_RANGES = [
    (
        54,
        80,
        "ألغيت المواد من 54 إلى 80 بالقرار الجمهوري.",
        "Articles 54-80 have been repealed by Presidential Decree.",
    ),
    (389, 417, "المواد من 389 إلى 417 ملغاة.", "Articles 389-417 are repealed."),
]


def extract_text_from_pdf(pdf_path):
    doc = pymupdf.open(pdf_path)
    return "\n".join(page.get_text() for page in doc)


LATIN_WORD = re.compile(r"[A-Za-z]{3,}")


def parse_text_to_json(full_text):
    heads = list(AR_HEADER.finditer(full_text))
    by_number = {}
    prev = 0

    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(full_text)
        block = full_text[h.end() : end]

        en = EN_HEADER.search(block)
        if en:
            number = int(en.group(1))
            rest = block[: en.start()] + "\n" + block[en.end() :]
        else:
            number = prev + 1
            rest = block
            print(f"[warn] no English header, assuming article {number}")

        first_latin = LATIN_WORD.search(rest)
        if first_latin:
            text_ar, text_en = rest[: first_latin.start()], rest[first_latin.start() :]
        else:
            text_ar, text_en = rest, ""

        text_ar = AR_STOP.split(text_ar)[0].strip()
        text_en = EN_STOP.split(text_en)[0].strip()
        if len(text_ar) < 10:
            print(f"[warn] article {number}: empty Arabic text, skipped")
            continue

        record = {
            "article_number": number,
            "text_ar": text_ar,
            "text_en": text_en,
            "is_repealed": False,
            "citation": f"Egyptian Civil Code, Article {number}",
        }
        if number in by_number:
            print(f"[warn] duplicate article {number}, keeping the longer one")
            if len(text_ar) <= len(by_number[number]["text_ar"]):
                continue
        by_number[number] = record
        prev = number

    for lo, hi, ar, en in REPEALED_RANGES:
        for n in range(lo, hi + 1):
            by_number[n] = {
                "article_number": n,
                "text_ar": ar,
                "text_en": en,
                "is_repealed": True,
                "citation": f"Egyptian Civil Code, Article {n}",
            }

    return [by_number[n] for n in sorted(by_number)]


def validate(articles):
    nums = [a["article_number"] for a in articles]
    assert len(nums) == len(set(nums)), "duplicate article numbers"
    missing = sorted(set(range(1, max(nums) + 1)) - set(nums))
    assert all(a["text_ar"].strip() for a in articles), "empty text_ar"
    assert all(len(a["text_ar"]) < 6000 for a in articles), "huge record = failed split"
    print(f"articles={len(nums)} max={max(nums)} missing={missing}")
    return missing


def save_to_json(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    print("Reading PDF...")
    text = extract_text_from_pdf(PDF_PATH)
    print("Parsing articles...")
    data = parse_text_to_json(text)
    validate(data)
    save_to_json(data, JSON_PATH)
    print(f"Success! {len(data)} articles saved to {JSON_PATH}")
