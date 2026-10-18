import json
import os
import shutil
from functools import lru_cache

from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_groq import ChatGroq
from langfuse.decorators import observe

from prodml import config

load_dotenv()

JSON_PATH = os.path.join(config.BASE_DIR, "data", "raw", "civil_code_articles.json")
CHROMA_PATH = os.path.join(config.BASE_DIR, "data", "chroma_db")

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
TOP_K = int(os.getenv("TOP_K", "5"))


SYSTEM_PROMPT = (
    "أنت مساعد قانوني متخصص في القانون المدني المصري. "
    "أجب باللغة العربية وبالاعتماد حصراً على المواد الواردة في السياق أدناه. "
    "اذكر رقم المادة التي استندت إليها بصيغة (المادة N). "
    "ميّز بدقة بين النص القانوني الصريح واستنتاجك منه، "
    "ولا تنسب إلى أي مادة شرطاً أو حكماً غير وارد في نصها. "
    "عند وجود عدة مواد ذات صلة، وضّح الشرط الخاص بكل مادة على حدة. "
    "إذا كانت المادة ملغاة فاذكر ذلك صراحة. "
    "إذا لم تجد الإجابة في السياق، فقل: "
    "لم أجد نصاً في المواد المتاحة يجيب عن هذا السؤال.\n\n"
    "السياق:\n{context}"
)


@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_db():
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embeddings())


@lru_cache(maxsize=1)
def get_chain():
    retriever = get_db().as_retriever(search_kwargs={"k": TOP_K})
    llm = ChatGroq(model_name=LLM_MODEL, temperature=0.1)
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", "{input}")]
    )
    doc_prompt = PromptTemplate.from_template("المادة {article_number}: {page_content}")
    qa_chain = create_stuff_documents_chain(llm, prompt, document_prompt=doc_prompt)
    return create_retrieval_chain(retriever, qa_chain)


def build_vector_db():
    print("Loading JSON data...")
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        articles = json.load(f)

    docs = [
        Document(
            page_content=a["text_ar"],
            metadata={
                "article_number": a["article_number"],
                "citation": a["citation"],
                "is_repealed": a.get("is_repealed", False),
            },
        )
        for a in articles
    ]

    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    for fn in (get_db, get_chain):
        fn.cache_clear()

    db = Chroma.from_documents(
        documents=docs, embedding=get_embeddings(), persist_directory=CHROMA_PATH
    )
    print(f"Indexed {len(docs)} articles with {EMBEDDING_MODEL}")
    return db


EMPTY_FALLBACK = "لم أجد نصاً في المواد المتاحة يجيب عن هذا السؤال."


@observe()
def ask_legal_question(question: str):
    result = get_chain().invoke({"input": question})
    if not result.get("answer", "").strip():
        result["answer"] = EMPTY_FALLBACK
    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build_vector_db()
    else:
        result = ask_legal_question("هل يجوز للقاضي تعديل العقد؟")
        print(result["answer"])
        for d in result["context"]:
            print(d.metadata["citation"])
