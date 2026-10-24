# Egyptian Civil Code RAG (MLOps Final Project, Track B)

A Retrieval-Augmented Generation system that answers questions in Arabic (and English) about the Egyptian Civil Code, always citing the article numbers it used. Built as the final project of the MLOps course, on top of the packaging, API, Docker, MLflow, DVC and CI work from Mini Projects 1 and 2.

> The previous project's README (NYC taxi predictor) is kept in `docs/README_mini_project_1.md`.

## Quickstart (3 commands)

Prerequisites: Docker, Python 3.10+ with the project venv (for `dvc`), and a `.env` file at the repo root (never committed). Create it with `cp .env.example .env` and fill in the values:

```
GROQ_API_KEY=...
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
EMBEDDING_MODEL=BAAI/bge-m3
```

```bash
# 1. Start the MinIO storage that holds the DVC data
docker compose -f docker/docker-compose.yml up -d minio createbucket

# 2. Pull the versioned data (PDF, parsed articles, Chroma vector store)
dvc pull

# 3. Build and start the RAG API on http://localhost:8001
docker compose -f docker/docker-compose.yml up -d --build rag
```

Try it:

```bash
curl localhost:8001/health
curl -s -X POST localhost:8001/ask -H "Content-Type: application/json" \
  -d '{"question":"ما هو سن الرشد في القانون المدني؟"}'
```

The response contains `answer`, `sources` (cited articles) and `latency_seconds`. The first start downloads the `bge-m3` embedding model (about 2 GB); the container reuses the host's Hugging Face cache (`~/.cache/huggingface`).

## How it works

```
PDF -> ingest.py -> civil_code_articles.json -> llm.py build -> Chroma (bge-m3)
                                                                  |
question -> FastAPI /ask -> retriever (top 5) -> Groq LLM -> answer + cited articles
                                   (every call traced in Langfuse)
```

- **Ingestion** (`src/prodml/ingest.py`): the PDF is bilingual and has reversed Arabic digits and broken lam-alef ligatures. The English "Article N" header is used as the article number, Arabic and English text are split, duplicates removed, repealed articles flagged, and a ligature repair step fixes common words. Result: 1133 articles.
- **Retrieval**: one document per article, embedded with `BAAI/bge-m3`, stored in Chroma, top 5 retrieved.
- **Generation**: Groq `openai/gpt-oss-20b`. The prompt forces citations ("المادة N") and an explicit "not found" answer when the context does not contain the answer.
- **API** (`src/prodml/api/rag_api.py`): FastAPI with input validation (422), a `/health` endpoint that reports how many documents are indexed (503 if the store is unavailable), and error messages that do not leak internals.
- **Observability**: Langfuse traces for each request and each evaluation run.

## Reproducibility

- **DVC pipeline** (`dvc.yaml`): `ingest` -> `build_index`, with the PDF, JSON and Chroma store versioned and pushed to the MinIO remote. Reproduce with `dvc repro`.
- **MLflow**: `egyptian_civil_rag_retrieval` (5 embedding/chunking configurations) and `egyptian_civil_rag_ragas`. Run locally with `MLFLOW_TRACKING_URI=sqlite:///mlflow.db`.
- **Docker**: `docker/Dockerfile.rag` (CPU torch, no models baked into the image).
- **CI** (`.github/workflows/ci.yml`): ruff + black, pytest with coverage gate (70%), then Docker build and push of both images.

## Evaluation

The evaluation set is `data/eval/questions.json`: 35 questions (32 answerable, including 3 in English; 1 about a repealed article; 2 out of scope).

## Load test (Locust)

`loadtest/locustfile.py`, 50 concurrent users, ramp 5 users/s, 90 s, against the Docker container on a CPU-only laptop, with the real Groq LLM. Report: `reports/locust_rag.html`.

| Endpoint | Requests | Failures | p50 | p95 | p99 |
|---|---|---|---|---|---|
| POST /ask | 122 | 108 (88.5%), all HTTP 500 | 26 s | 41 s | 47 s |
| GET /health | 17 | 0 | 0.11 s | 13 s | 13 s |

Result: **the service does not hold 50 concurrent users.** The first requests, before the load built up, completed in about 1.7 to 3.9 s. As load grew, most `/ask` calls failed with HTTP 500 and even `/health` slowed to 13 s at p95, so the whole service was saturated, not only the LLM call. I have not yet confirmed the root cause: candidates are the free Groq rate limit and CPU saturation from concurrent embedding calls. Not tried yet: a request queue or concurrency limit, retries with backoff, a smaller load level, a paid LLM key.

### Retrieval (no LLM involved)

Command: `python -m prodml.eval_retrieval`

| Configuration | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|
| bge-m3, whole articles (**chosen**) | 0.84 | 0.88 | 0.94 | 0.873 |
Embedding comparison (MLflow experiment `egyptian_civil_rag_retrieval`): `bge-m3` was clearly better than MiniLM and mpnet on this Arabic legal text, and splitting articles into chunks gave no improvement, so whole articles are indexed.

### Answer quality (RAGAS)

Command: `python -m prodml.eval` (judge: Groq `openai/gpt-oss-120b`).

Run on a 10-question subset (not the full 35).

| Metric | Mean (valid items) | NaN items |
|---|---|---|
| faithfulness | 0.64 | 4 / 10 |
| context_precision | 0.79 | 0 / 10 |
| context_recall | 1.00 | 0 / 10 |

Per-question scores: `reports/ragas_scores.csv`. Honest reading: faithfulness is **below the 0.75 target from the course handbook**. Many judge calls fail to parse (NaN) because the free Groq judge returns malformed JSON, so the mean is computed on valid items only and should be read with caution.

## Deviations from the course plan

- **Langfuse Cloud instead of a local Langfuse in Docker.** I used the hosted Langfuse (free tier) for tracing instead of running it in a local Docker container, to save setup time and memory on a laptop that already runs the embedding model.
- **Groq instead of a local LLM.** My GPU (RTX 3050, 4 GB) cannot be used by PyTorch with the installed driver, so everything runs on CPU. Generation and the RAGAS judge use Groq-hosted models.
- **Coverage gate with an omit list.** `llm.py`, `eval.py`, `eval_retrieval.py`, `experiments.py`, `ingest.py` and `api/rag_api.py` are excluded from the 70% coverage computation because they need external services or large models. The API has 3 tests with the LLM and the vector store mocked; the rest is covered by the evaluation scripts instead of unit tests.
- **The vector store is not baked into the Docker image.** It is versioned with DVC and mounted into the container (`data/chroma_db`), because the bge-m3 index is large and rebuilt by the DVC pipeline. The API listens on port 8001 (8000 is used by the Mini Project 1 API).

## Not done / future work

- Diagnose and fix the load-test bottleneck (confirm the cause, add a concurrency limit and retries).
- GPU optimisation and a self-hosted LLM (vLLM / AWQ quantisation).
- BentoML serving.
- Canary / staged rollout.
- Unit tests for `ingest.py` (`parse_text_to_json`, `fix_arabic_ligatures`).
- Direct lookup by article number (questions like "ما نص المادة 60؟") and a retrieval fallback for repealed articles.

## Known limitations

- 16 articles from the PDF are missing from the index (extraction problems).
- Ligature repair is partial: some words (for example "باطال", "خالف") are still corrupted.
- The model sometimes omits paragraphs of long articles (for example article 147, paragraph 2).
- The question about a repealed article (Article 60) is not answered correctly, because retrieval is semantic only.
- RAGAS faithfulness is unreliable with this judge (NaN items) and is below the target.
- This is an educational system, not legal advice.
- RAGAS was run on 10 questions only (the handbook asks for 50+ and all 4 metrics).
- Faithfulness mean (0.64) is below the 0.75 CI gate and below the 0.80 alert threshold in the handbook, so no automatic gate is enforced.
- The API does not hold 50 concurrent users (see Load test).
## Repository layout

```
src/prodml/        ingest.py, llm.py, eval.py, eval_retrieval.py, experiments.py, api/
data/eval/         questions.json
reports/           ragas_scores.csv
docker/            Dockerfile (taxi API), Dockerfile.rag, docker-compose.yml
tests/             API and serialization tests
dvc.yaml           ingest -> build_index
docs/              README of Mini Project 1
```
