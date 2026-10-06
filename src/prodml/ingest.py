import pymupdf
import json
import re
import os
from prodml import config

PDF_PATH = os.path.join(config.BASE_DIR, "data", "raw", "egyptian_civil_code.pdf")
JSON_PATH = os.path.join(config.BASE_DIR, "data", "raw", "civil_code_articles.json")


def extract_text_from_pdf(pdf_path):
    full_text = ""
    doc = pymupdf.open(pdf_path)
    for page in doc:
        full_text += page.get_text()
    return full_text


# data in json form
def parse_text_to_json(full_text):
    articles_list = []

    chunks = re.split(r"(مادة\s*\d+|Article\s*\d+)", full_text)

    for i in range(1, len(chunks), 2):
        if i + 1 < len(chunks):
            header = chunks[i]
            content = chunks[i + 1].strip()

            num_match = re.search(r"\d+", header)
            if num_match:
                actual_num = int(num_match.group())
            else:
                continue

            if len(content) < 20:
                continue

            article_dict = {
                "article_number": actual_num,
                "text_ar": content,
                "citation": f"Egyptian Civil Code, Article {actual_num}",
            }
            articles_list.append(article_dict)

    return articles_list


def save_to_json(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    print("Reading PDF...")
    text = extract_text_from_pdf(PDF_PATH)

    print("Parsing articles...")
    structured_data = parse_text_to_json(text)

    print("Saving to JSON...")
    save_to_json(structured_data, JSON_PATH)

    print(f"Success! {len(structured_data)} articles saved to {JSON_PATH}")
