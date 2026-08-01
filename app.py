import os
from threading import Lock

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.exceptions import OutputParserException
from langchain_core.globals import set_llm_cache
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from openai import AuthenticationError

from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt

app = Flask(__name__)
load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not PINECONE_API_KEY or "your" in PINECONE_API_KEY.lower():
    raise RuntimeError("PINECONE_API_KEY is missing or a placeholder. Please set it in your .env file.")
if not OPENROUTER_API_KEY or "your" in OPENROUTER_API_KEY.lower():
    raise RuntimeError("OPENROUTER_API_KEY is missing or a placeholder. Please set it in your .env file.")

set_llm_cache(None)

rag_chain = None
_lock = Lock()


def create_rag_chain():
    embeddings = download_hugging_face_embeddings()
    index_name = os.environ.get("INDEX_NAME", "medical-chatbot")
    docsearch = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    chat_model = ChatOpenAI(
        model="openai/gpt-4o-mini",
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        temperature=0,
    )
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
    combine_docs_chain = create_stuff_documents_chain(chat_model, prompt)
    return create_retrieval_chain(retriever, combine_docs_chain)


def get_rag_chain():
    global rag_chain
    if rag_chain is None:
        with _lock:
            if rag_chain is None:
                rag_chain = create_rag_chain()
    return rag_chain


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/health")
def health():
    return jsonify({"status": "healthy"})


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form.get("msg", "").strip()
    if not msg:
        return jsonify({"error": "Please enter a question."}), 400

    try:
        chain = get_rag_chain()
        response = chain.invoke({"input": msg})
        answer = response.get("answer", "Sorry, I could not find an answer.")
        return jsonify({"answer": answer})
    except AuthenticationError:
        return jsonify({"error": "Authentication failed. Please check OPENROUTER_API_KEY."}), 500
    except OutputParserException:
        return jsonify({"error": "Sorry, I had trouble formatting the response."}), 500
    except Exception:
        return jsonify({"error": "An unexpected error occurred. Please check the server logs."}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
