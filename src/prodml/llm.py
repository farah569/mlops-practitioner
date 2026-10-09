import os
import json
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from prodml import config

load_dotenv()

JSON_PATH = os.path.join(config.BASE_DIR, "data", "raw", "civil_code_articles.json")
CHROMA_PATH = os.path.join(config.BASE_DIR, "data", "chroma_db")


def build_vector_db():
    print("Loading JSON data...")
    with open(JSON_PATH, "r", encoding="utf-8") as file:
        articles = json.load(file)

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

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    vector_db = Chroma.from_documents(
        documents=docs, embedding=embeddings, persist_directory=CHROMA_PATH
    )
    return vector_db


def ask_legal_question(question: str):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    retriever = db.as_retriever(search_kwargs={"k": 3})

    llm = ChatGroq(model_name="openai/gpt-oss-20b", temperature=0.1)

    system_prompt = (
        "أنت محامي مصري خبير في القانون المدني. "
        "استخدم السياق التالي المقتبس من القانون المدني للإجابة على سؤال المستخدم باللغة العربية.\n\n"
        "السياق:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    print(f"\nThinking about: '{question}'...\n")
    response = rag_chain.invoke({"input": question})

    return response


if __name__ == "__main__":
    my_question = "ما هي شروط إبطال العقد؟"

    result = ask_legal_question(my_question)

    print("\n--- الإجابة ---")
    print(result["answer"])

    print("\n--- المصادر اللي جاب منها الإجابة ---")
    for doc in result["context"]:
        print(f"- {doc.metadata['citation']}")
