"""RAGAS evaluation over data/eval/questions.json."""

import argparse
import json
import math
import os
import time

import mlflow
from datasets import Dataset
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langfuse import Langfuse
from ragas import evaluate
from ragas.metrics import context_precision, context_recall, faithfulness
from ragas.run_config import RunConfig

from prodml import config
from prodml.llm import ask_legal_question

load_dotenv()

QUESTIONS_PATH = os.path.join(config.BASE_DIR, "data", "eval", "questions.json")
REPORT_PATH = os.path.join(config.BASE_DIR, "reports", "ragas_scores.csv")
METRIC_NAMES = ["faithfulness", "context_precision", "context_recall"]

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-oss-20b")
JUDGE_EFFORT = os.getenv("JUDGE_EFFORT", "medium")


def load_questions(ci_only, limit):
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        items = [q for q in json.load(f) if q["category"] == "answerable"]
    if ci_only:
        items = [q for q in items if q["ci"]]
    return items[:limit] if limit else items


def run_evaluation(ci_only=False, limit=None):
    items = load_questions(ci_only, limit)
    print(
        f"Evaluating {len(items)} questions (judge={JUDGE_MODEL}, effort={JUDGE_EFFORT})"
    )

    data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for i, item in enumerate(items, 1):
        print(f"[{i}/{len(items)}] {item['question']}")
        result = ask_legal_question(item["question"])
        data["question"].append(item["question"])
        data["answer"].append(result["answer"])
        data["contexts"].append([d.page_content for d in result["context"]])
        data["ground_truth"].append(item["ground_truth"])
        time.sleep(3)

    judge_llm = ChatGroq(
        model_name=JUDGE_MODEL,
        temperature=0.0,
        model_kwargs={"reasoning_effort": JUDGE_EFFORT},
    )
    judge_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    print("\nGrading (this takes a while)...\n")
    score = evaluate(
        dataset=Dataset.from_dict(data),
        metrics=[faithfulness, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=RunConfig(max_workers=1, max_retries=10, timeout=180),
    )

    df = score.to_pandas()
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    df.to_csv(REPORT_PATH, index=False, encoding="utf-8-sig")

    print("\nScorecard (mean ignores NaN):")
    for m in METRIC_NAMES:
        n_nan = int(df[m].isna().sum())
        print(f"  {m:18s} mean={df[m].mean():.3f}   NaN={n_nan}/{len(df)}")
    print(f"\nPer-question scores saved to {REPORT_PATH}")

    lf = Langfuse()
    for m in METRIC_NAMES:
        mean_val = df[m].mean()
        if not math.isnan(mean_val):
            lf.score(name=f"ragas_{m}", value=float(mean_val))
    lf.flush()
    mlflow.set_experiment("egyptian_civil_rag_ragas")
    with mlflow.start_run(run_name="ragas-bge-m3-whole"):
        mlflow.log_params(
            {
                "judge_model": JUDGE_MODEL,
                "judge_effort": JUDGE_EFFORT,
                "n_questions": len(df),
                "embedding_model": os.getenv("EMBEDDING_MODEL", "default"),
            }
        )
        for m in METRIC_NAMES:
            mean_val = df[m].mean()
            if not math.isnan(mean_val):
                mlflow.log_metric(m, float(mean_val))
            mlflow.log_metric(f"{m}_nan", int(df[m].isna().sum()))
        mlflow.log_artifact(REPORT_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ci", action="store_true", help="only the 20 CI questions")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run_evaluation(ci_only=args.ci, limit=args.limit)
