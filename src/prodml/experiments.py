"""Retrieval experiments logged to MLflow.

Usage:
    python -m prodml.experiments              # all 5 configs (slow)
    python -m prodml.experiments --only 0,3   # a subset, by index
"""

import argparse
import json
import os
import time
import uuid

import mlflow
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from prodml import config

load_dotenv()

ARTICLES_PATH = os.path.join(config.BASE_DIR, "data", "raw", "civil_code_articles.json")
QUESTIONS_PATH = os.path.join(config.BASE_DIR, "data", "eval", "questions.json")

MINILM = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MPNET = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
BGE = "BAAI/bge-m3"

CONFIGS = [
    {"embedding_model": MINILM, "chunk_size": 0, "chunk_overlap": 0},
    {"embedding_model": MINILM, "chunk_size": 500, "chunk_overlap": 50},
    {"embedding_model": MPNET, "chunk_size": 0, "chunk_overlap": 0},
    {"embedding_model": BGE, "chunk_size": 0, "chunk_overlap": 0},
    {"embedding_model": BGE, "chunk_size": 500, "chunk_overlap": 50},
]
TOP_K = 5


def build_docs(articles, chunk_size, chunk_overlap):
    splitter = None
    if chunk_size:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    docs = []
    for a in articles:
        parts = splitter.split_text(a["text_ar"]) if splitter else [a["text_ar"]]
        for part in parts:
            docs.append(
                Document(
                    page_content=part,
                    metadata={"article_number": a["article_number"]},
                )
            )
    return docs


def ranked_articles(db, question):
    """Top-TOP_K distinct article numbers (chunks of the same article are merged)."""
    seen = []
    for d in db.similarity_search(question, k=TOP_K * 4):
        n = d.metadata["article_number"]
        if n not in seen:
            seen.append(n)
        if len(seen) == TOP_K:
            break
    return seen


def run_one(cfg, articles, questions):
    short = cfg["embedding_model"].split("/")[-1]
    run_name = f"{short}-chunk{cfg['chunk_size']}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({**cfg, "top_k": TOP_K, "ligature_fix": True})

        t0 = time.time()
        docs = build_docs(articles, cfg["chunk_size"], cfg["chunk_overlap"])
        embeddings = HuggingFaceEmbeddings(model_name=cfg["embedding_model"])
        db = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            collection_name=f"exp-{uuid.uuid4().hex[:12]}",  # fresh collection per run
        )  # in-memory
        index_seconds = time.time() - t0

        hits = {1: 0, 3: 0, 5: 0}
        rr = 0.0
        for q in questions:
            ranks = ranked_articles(db, q["question"])
            if q["gold_article"] in ranks:
                pos = ranks.index(q["gold_article"]) + 1
                rr += 1 / pos
                for k in hits:
                    if pos <= k:
                        hits[k] += 1

        n = len(questions)
        metrics = {
            "hit_at_1": hits[1] / n,
            "hit_at_3": hits[3] / n,
            "hit_at_5": hits[5] / n,
            "mrr": rr / n,
            "n_chunks": len(docs),
            "index_seconds": index_seconds,
        }
        mlflow.log_metrics(metrics)
        print(run_name, {k: round(v, 3) for k, v in metrics.items()})


def main(only):
    with open(ARTICLES_PATH, encoding="utf-8") as f:
        articles = json.load(f)
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = [q for q in json.load(f) if q["category"] == "answerable"]

    mlflow.set_experiment("egyptian_civil_rag_retrieval")
    for i, cfg in enumerate(CONFIGS):
        if only is None or i in only:
            print(f"\n=== config {i}: {cfg}")
            run_one(cfg, articles, questions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None, help="e.g. 0,3")
    args = parser.parse_args()
    only = {int(x) for x in args.only.split(",")} if args.only else None
    main(only)
