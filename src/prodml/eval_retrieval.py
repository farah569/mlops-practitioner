"""Retrieval-only evaluation (no judge LLM, instant, deterministic)."""

import argparse
import json
import os

from prodml import config
from prodml.llm import TOP_K, ask_legal_question, get_db

QUESTIONS_PATH = os.path.join(config.BASE_DIR, "data", "eval", "questions.json")


def main(behavior):
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)

    db = get_db()
    answerable = [q for q in questions if q["category"] == "answerable"]
    hits = {1: 0, 3: 0, TOP_K: 0}
    rr_sum = 0.0
    misses = []

    for q in answerable:
        docs = db.similarity_search(q["question"], k=TOP_K)
        ranks = [d.metadata["article_number"] for d in docs]
        if q["gold_article"] in ranks:
            rank = ranks.index(q["gold_article"]) + 1
            rr_sum += 1 / rank
            for k in hits:
                if rank <= k:
                    hits[k] += 1
        else:
            misses.append((q["gold_article"], q["question"], ranks))

    n = len(answerable)
    print(f"Questions: {n}  (embedding/top_k come from .env)")
    for k in sorted(hits):
        print(f"  hit@{k}: {hits[k]}/{n} = {hits[k] / n:.2f}")
    print(f"  MRR: {rr_sum / n:.3f}")
    if misses:
        print("\nMisses (gold not in top-k):")
        for gold, question, ranks in misses:
            print(f"  gold={gold} got={ranks} | {question}")

    if behavior:
        print("\n--- behavior checks (read the answers yourself) ---")
        for q in questions:
            if q["category"] in ("repealed", "out_of_scope"):
                print(f"\n[{q['category']}] {q['question']}")
                print(ask_legal_question(q["question"])["answer"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--behavior", action="store_true")
    main(parser.parse_args().behavior)
