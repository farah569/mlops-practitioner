import json
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from prodml import config

JSON_PATH = os.path.join(config.BASE_DIR, "data", "raw", "civil_code_articles.json")
CHROMA_PATH = os.path.join(config.BASE_DIR, "data", "chroma_db")


def build_vector_db():
    print("Loading JSON data...")

    with open(JSON_PATH, "r", encoding="utf-8") as file:
        articles = json.load(file)

    print("Preparing documents...")
    docs = []
    for article in articles:
        doc = Document(
            page_content=article["text_ar"],
            metadata={
                "article_number": article["article_number"],
                "citation": article["citation"],
            },
        )
        docs.append(doc)

    print("Downloading embedding model (this understands Arabic!)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    print("Building Vector Database (this might take a minute)...")

    vector_db = Chroma.from_documents(
        documents=docs, embedding=embeddings, persist_directory=CHROMA_PATH
    )

    print(f"Success! Vector database built at {CHROMA_PATH}")
    return vector_db


if __name__ == "__main__":
    build_vector_db()
